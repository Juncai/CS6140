import DataLoader as loader

DATA = 'data/'
CRX_CONFIG = DATA + 'crx/crx.config'
CRX_DATA = DATA + 'crx/crx.data'
CRX_OUT_PATH = DATA + 'crx_parsed.data'
VOTE_CONFIG = DATA + 'vote/vote.config'
VOTE_DATA = DATA + 'vote/vote.data'
VOTE_OUT_PATH = DATA + 'vote_parsed.data'

# load CRX data
crx_meta = loader.parse_UCI_config(CRX_CONFIG)
loader.load_UCI_data(CRX_DATA, crx_meta, CRX_OUT_PATH)

# load VOTE data
vote_meta = loader.parse_UCI_config(VOTE_CONFIG)
loader.load_UCI_data(VOTE_DATA, vote_meta, VOTE_OUT_PATH)