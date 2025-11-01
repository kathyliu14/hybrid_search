# project settings
from pathlib import Path
# project wise settings
SEED = 6400
N = 150_000

# for data loading
REPO_ID = "DragonLLM/Clean-Wikipedia-English-Articles"
CHUNK_FETCH = 50_000     # fetch this many, shuffle locally, then write
LOG_EVERY = 2_000
OUT_CSV = "wikipedia_sample_150k_with_ids.csv"
COLUMNS_CSV = ["item_id","id","title","text","url","revdate","token_count","entity"]

# for csv loading
CHUNK_SIZE = 10_000


# for sqlite database creation
DB_PATH = Path("meta_wiki.db")
USECOLS = ["item_id", "id", "title", "url", "revdate", "token_count", "entity"]
ITEM_COLS_CSV = ["item_id", "id", "title", "url", "revdate", "token_count", "entity"]
ITEM_COLS_DB = ["item_id", "ext_id", "title", "url", "revdate", "token_count", "entity"]
PRIMARY_KEY = "item_id"
TABLE_NAME = "items"
DATE_COLUMNS = ["revdate"]
INT_COLUMNS_CSV = ["item_id", "id", "token_count"]
INT_COLUMNS_DB = ["item_id", "ext_id", "token_count"]
# for now, we try to only implement two predicates, one for numeric and one for date. 
# later we can try categories and lexicographic predicates
PREDICATE_COLUMNS = ["item_id", "revdate", "token_count"]


# IVF settings
NLIST        = 512                 # #clusters ~ sqrt(N)
PQ_M         = 32                   # #subquantizers
PQ_BITS      = 8                    # bits per subquantizer
TRAIN_MAX    = 60_000               # max vectors to train on (sampled)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COL = "text"
ITEM_COL_ID = "item_id"
FAISS_PATH = "index.faiss.new"
NPROBE = 16
N_PER_CLUSTER = 30_000
# for embedding 
EMBED_BATCH = 64

# for search
DISPLAY_COLS = ["item_id", "title","revdate", "token_count"]

COLUMNS = ["title", "text"]