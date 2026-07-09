TATR_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"
TATR_THRESHOLD = 0.5
TATR_RESIZE = 800

TABLE_LABEL = "table"

MARGIN_X = 50
MARGIN_Y = 30
MARGIN_X_SMALL_CLUSTER = 200
MARGIN_Y_SMALL_CLUSTER = 80
SMALL_CLUSTER_THRESHOLD = 8

VERTICAL_GAP_MULTIPLIER = 3.0
COLUMN_GAP_RATIO = 0.05
EXPAND_MARGIN = 20
MAX_TABLE_REGION_AREA_RATIO = 0.50
MIN_TOKENS_PER_REGION = 5
MIN_ROWS = 2
MIN_COLS = 1
MAX_REGION_AREA_RATIO = 1.0

PARAGRAPH_MIN_WORDS = 12
PARAGRAPH_MAX_WORDS_NO_NUMERIC = 5

ESG_KEYWORDS = [
    "scope 1", "scope 2", "scope 3",
    "emission", "ghg", "co2", "co2e", "tco2e", "tco2",
    "energy", "electricity", "fuel", "power",
    "phat thai", "khi nha kinh", "nang luong",
    "carbon", "greenhouse gas", "climate",
]
ESG_UNIT_PATTERNS = [
    r"tCO2e", r"tCO[₂2]e?", r"tCO2",
    r"kgCO2e", r"kgCO[₂2]e",
    r"gCO2e", r"gCO[₂2]e",
    r"MWh", r"kWh", r"GWh",
    r"VND", r"USD", r"EUR", r"%",
]
YEAR_PATTERN = r"(FY|CY|FY_)?(20\d{2})"
