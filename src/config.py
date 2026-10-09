"""
Configuration and Lexical Resources for Tagalog / Taglish NLP Pipeline.
Includes Tagalog and English stop words, slang dictionaries, domain stop words,
and iterative sentiment lexicons.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw" / "sample_dataset.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# -------------------------------------------------------------
# Stop Words Lists
# -------------------------------------------------------------
TAGALOG_STOPWORDS = {
    'ang', 'mga', 'sa', 'ng', 'na', 'si', 'ni', 'kay', 'at', 'o', 'pero', 
    'dahil', 'kasi', 'para', 'kung', 'kapag', 'nang', 'ay', 'ito', 'iyan', 
    'iyon', 'dito', 'diyan', 'doon', 'nito', 'niyan', 'noon', 'ako', 'ikaw', 
    'ka', 'siya', 'kami', 'tayo', 'kayo', 'sila', 'ko', 'mo', 'niya', 'namin', 
    'natin', 'ninyo', 'nila', 'akin', 'iyo', 'kaniya', 'amin', 'atin', 'inyo', 
    'kanila', 'may', 'mayroon', 'wala', 'ba', 'eh', 'naman', 'pa', 'din', 
    'rin', 'raw', 'daw', 'po', 'opo', 'nga', 'pala', 'lang', 'lamang', 'sana', 
    'baga', 'man', 'yata', 'kundi', 'habang', 'mula', 'hanggang', 'bago', 
    'pagkatapos', 'upang', 'kahit', 'bagaman', 'subalit', 'datapwat', 'un', 'ung',
    'yan', 'yun', 'eto', 'sya', 'nyo', 'nla', 'dina', 'jan', 'dn', 'oh', 'pla', 'ano',
    'naging', 'dinig', 'nandyan', 'nandoon', 'nandito', 'yung', 'nung', 'nya', 'kaya',
    'hahaha', 'haha', 'dapat', 'talaga', 'nag', 'lahat', 'sir', 'hindi', 'di', 'walang'
}

ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
    'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
}

DOMAIN_STOPWORDS = {
    'amp', 'rt', 'via', 'http', 'https', 'com', 'co', 'twitter', 'post', 'status',
    'abscbnnews', 'news', 'update', 'repost', 'pls', 'thank', 'thanks'
}

ALL_STOPWORDS = TAGALOG_STOPWORDS.union(ENGLISH_STOPWORDS).union(DOMAIN_STOPWORDS)

# -------------------------------------------------------------
# Tagalog Social Media Slang & Contractions
# -------------------------------------------------------------
TAGALOG_SLANG_DICT = {
    'un': 'iyon',
    'ung': 'iyong',
    'sya': 'siya',
    'eto': 'ito',
    'yan': 'iyan',
    'yun': 'iyon',
    'bec': 'because',
    'bcoz': 'because',
    'pls': 'please',
    'lng': 'lang',
    'wag': 'huwag',
    'di': 'hindi',
    'd': 'hindi',
    'nman': 'naman',
    'pla': 'pala',
    'kht': 'kahit',
    'bkt': 'bakit',
    'aq': 'ako',
    'ikw': 'ikaw',
    'trapos': 'trapo',
    'sh*tshow': 'shitshow',
    'dutaes': 'duterte',
    'tsina': 'china',
    'pax': 'people',
    'kase': 'kasi'
}

# -------------------------------------------------------------
# Sentiment Lexicons (Iterative Improvement Requirement)
# -------------------------------------------------------------
# Iteration 0: Baseline Lexicon (General English + Basic Tagalog)
SENTIMENT_LEXICON_V1 = {
    'positive': {
        'good', 'great', 'excellent', 'happy', 'love', 'blessed', 'support',
        'maganda', 'maayos', 'tama', 'katotohanan', 'tindig', 'salamat', 'mabuti'
    },
    'negative': {
        'bad', 'worst', 'terrible', 'hate', 'corrupt', 'burn', 'hell',
        'basura', 'masama', 'nakakainis', 'galit', 'mali', 'dismaya'
    }
}

# Iteration 1: Domain-Specific Expansion (Political terms, Tagalog slangs, and Demojized emotions)
SENTIMENT_LEXICON_V2 = {
    'positive': SENTIMENT_LEXICON_V1['positive'].union({
        'accountability', 'transparency', 'justice', 'rights', 'truth', 'principles',
        'erudite', 'honest', 'pag-asa', 'prinsipyo', 'galing', 'bilib', 'respeto',
        'thumbs_up', 'clapping_hands', 'folded_hands', 'sparkles', 'check_mark_button'
    }),
    'negative': SENTIMENT_LEXICON_V1['negative'].union({
        'demonyo', 'kanser', 'kumag', 'tuso', 'trapo', 'tae', 'magnanakaw', 'pahirapan',
        'kabaluktutan', 'kadiliman', 'kasamaan', 'nakakagigil', 'nakakadismaya', 'nakakahiya',
        'nakakapagod', 'halatado', 'spineless', 'opportunists', 'tangina', 'tanginang',
        'bullshit', 'shitshow', 'malversation', 'guilt', 'delayed', 'incompetent',
        'enraged_face', 'angry_face', 'face_with_symbols_on_mouth', 'pile_of_poo',
        'lying_face', 'clown_face', 'broken_heart', 'crocodile'
    })
}

# Negation and Modifier terms
NEGATION_WORDS = {
    'hindi', 'di', 'wala', 'walang', 'ayaw', 'huwag', 'wag', 'never', 'not', 
    'no', 'without', 'neither', 'nor'
}

INTENSIFIERS = {
    'sobra', 'sobrang', 'masyado', 'masyadong', 'napaka', 'tunay', 'very', 
    'extremely', 'super', 'talaga', 'talagang', 'puro', 'purong'
}
