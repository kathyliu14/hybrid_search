import faiss  # do not remove this import, there is an apple silicon issue with faiss's importing order
from dataclasses import dataclass
from typing import Optional, List, Literal
import numpy as np
from vector_index import Embedder, FaissIVFIndex, AnnSearchResults
from settings import FAISS_PATH, NPROBE, NLIST, N
from dbManagement import DbManagement, DBRecord, DBRecords
from shared_dataclasses import Predicate
from histo2d import Histo2D
import pandas as pd
from settings import N_PER_CLUSTER

@dataclass
class HSearchResult:
    record: DBRecord
    similarity: float

@dataclass
class HSearchResults:
    results: List[HSearchResult]
    is_k: bool

    def to_df(self, show_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df = DBRecords([result.record for result in self.results]).to_df()
        if show_cols is not None:
            df = df.loc[:, show_cols]
        df["similarity"] = [result.similarity for result in self.results] # always show similarity
        return df.sort_values(by="similarity", ascending=False)


class Search:
    def __init__(self):
        self.index = FaissIVFIndex.load(FAISS_PATH)
        self.embedder = Embedder()
        self.db = DbManagement()
        self.histo = Histo2D.from_records(self.db.predicates_search([]))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    def close(self):
        """Close search resources."""
        self.db.close()


    def search(self, query: str, predicates: List[Predicate], k: int, method: Literal["pre_search", "post_search"] = "post_search") -> HSearchResults:
        embedding_query = self.embedder.encode_query(query)
        est_survivors = self.histo.estimate_survivors(predicates)
        if method == "pre_search":
            return self.pre_search(embedding_query, predicates, k)
        elif method == "post_search":
            return self.pos_search(embedding_query, predicates, k, est_survivors)
        else:
            raise ValueError(f"Invalid method: {method}")

    def pre_search(self, embedding_query: np.ndarray, predicates: List[Predicate], k: int) -> HSearchResults:
        filtered_records = self.db.predicates_search(predicates)
        
        if len(filtered_records) == 0:
            return HSearchResults(results=[], is_k=False)
        
        item_ids = [record.item_id for record in filtered_records.records]
        
        ann_results = self.index.search(embedding_query, max(k, len(filtered_records)), nprobe=NLIST, item_ids=item_ids)
        
        results_with_similarity = self._intersect(ann_results, filtered_records)
        
        if len(results_with_similarity) >= k:
            return HSearchResults(results=results_with_similarity[:k], is_k=True)
        return HSearchResults(results=results_with_similarity, is_k=False)
    
    def pos_search(self, embedding_query: np.ndarray, predicates: List[Predicate], k: int, est_survivors: int) -> HSearchResults:
        # we need to implement coarse post-search here
        return self.adaptive_pos_search(embedding_query, predicates, k, est_survivors)

    def adaptive_pre_search(self, embedding_query: np.ndarray, predicates: List[Predicate], k: int) -> HSearchResults:
        pre_db_records = self.db.predicates_search(predicates)
        pre_predicate_count = len(pre_db_records)
        if pre_predicate_count == 0:
            return HSearchResults(results=[], is_k=False)
        pre_item_ids = [record.item_id for record in pre_db_records.records]

        nprobe = 0
        ann_results = AnnSearchResults.empty()
        iteration = 0
        while len(ann_results) < k:
            print(f"------------START OF ITERATION {iteration + 1} OF PRE-SEARCH-------------")
            print(f"len(ann_results) = {len(ann_results)}, k = {k}")
            print(f"nprobe = {nprobe}")
            nprobe = self._opt_pre_nprobe(pre_predicate_count, k - len(ann_results), nprobe)
            ann_results = self.index.search(embedding_query, k, nprobe=nprobe, item_ids=pre_item_ids)
            iteration += 1
        pre_top_results = self._intersect(ann_results, pre_db_records)
        return HSearchResults(results=pre_top_results[:k], is_k=len(pre_top_results) >= k)
    
    def adaptive_pos_search(self, embedding_query: np.ndarray, predicates: List[Predicate], k: int, est_survivors: int) -> HSearchResults:
        top_results = []
        prev_search_k = 0
        iteration = 0

        while len(top_results) < k:
            print(f"------------START OF ITERATION {iteration + 1} OF POST-SEARCH-------------")
            # Get next search parameters with aggressive growth
            search_k, nprobe = self._opt_pos_search_k_and_nprobe(est_survivors, k - len(top_results), prev_search_k)
            print(f"search_k = {search_k}, nprobe = {nprobe}")
            # If we've reached the full dataset, search everything
            if search_k >= N:
                search_k = N
                nprobe = NLIST

            ann_results = self.index.search(embedding_query, search_k, nprobe=nprobe, item_ids=None)
            item_ids_predicate = Predicate(key="item_id", value=ann_results.item_ids.tolist(), operator="IN")
            predicates_copy = predicates.copy()
            predicates_copy.append(item_ids_predicate)
            post_db_records = self.db.predicates_search(predicates_copy)
            post_top_results = self._intersect(ann_results, post_db_records)
            top_results.extend(post_top_results)

            # Break if we can't make progress (no new results found)
            if len(post_top_results) == 0 and search_k >= N:
                break

            prev_search_k = search_k
            iteration += 1

        # Return exactly k results if possible, otherwise return all found
        final_results = top_results[:k] if len(top_results) >= k else top_results
        is_k_exact = len(final_results) == k
        return HSearchResults(results=final_results, is_k=is_k_exact)

    def _opt_pre_nprobe(self, predicate_count: int, k_remaining: int, old_nprobe: int = 0) -> int:
        """
        Find the optimal nprobe for the query and predicates.
        """
        # each cluster is expected to have predicate_count / NLIST survivors
        # the expected number of clusters to search is k / (predicate_count / NLIST)
        expected_nprobe = k_remaining * NLIST // predicate_count + old_nprobe
        # Use 3x multiplier for safety to ensure we get enough results
        return max(3, min(3 * expected_nprobe, NLIST))
    
    def _opt_pos_search_k_and_nprobe(self, est_survivors: int, k_remaining: int, old_search_k: int = 0, old_nprobe: int = 0):
        """
        Find the optimal search_k and nprobe for the query and est_survivors.
        Aggressive exponential growth to reach N quickly.
        """
        if est_survivors == 0:
            # If no survivors expected, search the entire dataset
            search_k = N
            nprobe = NLIST
        else:
            # Much more aggressive exponential growth
            if old_search_k == 0:
                # Start with a reasonable initial search based on selectivity
                if est_survivors >= N * 0.1:  # Low selectivity
                    base_search_k = N // 10  # Start with 10% of dataset
                else:  # High selectivity
                    base_search_k = max(k_remaining * 10, int(N / est_survivors * k_remaining * 5))
            else:
                # Exponential growth: 3x the previous size to converge quickly
                base_search_k = old_search_k * 3

            search_k = min(N, base_search_k)
            nprobe = min(NLIST, max(1, search_k // N_PER_CLUSTER * 4 + old_nprobe + 1))
        return search_k, nprobe


    @staticmethod
    def _intersect(ann_results: AnnSearchResults, db_records: DBRecords) -> List[HSearchResult]:
        db_records_dict = db_records.to_dict()
        db_item_ids = set(db_records_dict.keys())
        ann_results_dict = ann_results.to_dict()

        return [HSearchResult(record=db_records_dict[item_id], similarity=similarity)
                for item_id, similarity in ann_results_dict.items()
                if item_id in db_item_ids]
    

