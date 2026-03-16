"""
Geometry Project Two - Language Convex Hulls
Authors: Eden T, Luke F, Jacob R
Process: Word(string) --> TextPath --> Convex Hull

Each word is rendered as font geometry using matplotlib's TextPath. We then
compute the convex hull of those outlines and extract shape features.
Different scripts (Latin, Arabic, Chinese) are expected to produce geometrically 
distinct hulls.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from scipy.spatial import ConvexHull
import nltk

FONT_PATHS = {
    "English": "fonts/Noto_Sans/static/NotoSans_SemiCondensed-Regular.ttf",
    "Arabic": "fonts/Noto_Sans_Arabic/static/NotoSansArabic_SemiCondensed-Regular.ttf",
    "Chinese":  "fonts/Noto_Sans_TC/static/NotoSansTC-Regular.ttf",
}

COLORS = {
    "English": "#4A90D9",
    "Chinese": "#E85D4A",
    "Arabic":  "#50C878",
}

def load_english_words(n: int) -> list[str]:
    """
    Pull the n most common words from NLTK's Brown Corpus.
    Filters to alphabetic-only words to avoid punctuation tokens.
    """
    nltk.download('brown', quiet=True)
    from nltk.corpus import brown
    all_words = [w.lower() for w in brown.words() if w.isalpha()]
    unique_words = list(dict.fromkeys(all_words))
    return unique_words[:n]

def load_words_from_frequency_file(filepath: str, n: int) -> list[str]:
    """
    Load the top n words from a frequency word list file.
    Each line in the file is: "word count" — we just want the word.
    for the Chinese and Arabic frequency files.
    """
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    return [line.split()[0] for line in lines if line.strip()][:n]

def load_word_lists(n_per_language: int) -> dict:
    """
    Load n words for each language and return as the WORDS dict.
    """
    return {
        "English": load_english_words(n_per_language),
        "Chinese": load_words_from_frequency_file("zh_tw_50k.txt", n_per_language),
        "Arabic":  load_words_from_frequency_file("ar_50k.txt",    n_per_language),
    }

def load_font(lang: str) -> FontProperties: 
    """
        Font Configuration object, describes how text should
        look when rendered. Points directly to a .ttf file.
    """
    return FontProperties(fname=FONT_PATHS[lang])

def build_font_registry(languages: list) -> dict:
    """
        Build a {language -> FontProperties} dictionary for all
        Prevents any need to reload fonts for every word.
    """
    return {lang: load_font(lang) for lang in languages}

def render_text_path(word:str, font: FontProperties, size:int) -> np.ndarray:
    """
        Use matplotlib's textpath to render a word as a 2d path and return
        the (x,y) points sampled along the actual letter edges, curves are flattened
        fire, so every point in the returned (N,2) array lies on the visable boundary
        of the text.
    """
    path = TextPath((0,0), word, size=size, prop=font)
    return path.vertices

def is_anchor_point(point: np.ndarray) -> bool:
    """
        Return True if a point is an anchor point, this means a point in which the text
        path has to pick up the pen and "reset" so it inserts (0,0) as an anchor point. 
        These are not part of the letter and would skew the convex hull
    """
    #point[0] == 0 , x is exactly 0
    #point[1] == 0 , y is exactly 0
    return point[0] == 0 and point[1] == 0

def filter_anchor_points(verts: np.ndarray) -> np.ndarray:
    """
    Remove all artifact (0, 0) anchor points from a vertex array.
    Returns only the points that represent real glyph geometry.
    """
    mask = np.array([not is_anchor_point(p) for p in verts])
    return verts[mask]

def compute_convex_hull(points: np.ndarray):
    """
    Compute the convex hull for a set of 2D points.
    """
    if len(points) < 3:
        return None
    return ConvexHull(points)


def plot_word_hull(ax, points, hull, color, title,font):
    """
    Plot sampled text points and their convex hull.
    """
    ax.scatter(points[:,0], points[:,1], s=1, color="black", alpha=0.4)

    if hull is not None:
        hull_pts = points[hull.vertices]
        polygon = Polygon(hull_pts, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(polygon)

    ax.set_title(title, fontproperties=font)
    ax.set_aspect("equal")
    ax.axis("off")


def visualize_language(language, words, font):
    """
    Plot convex hulls for all words in a language.
    """
    n = len(words)
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten()

    for i, word in enumerate(words):

        verts = render_text_path(word, font, size=100)
        verts = filter_anchor_points(verts)

        hull = compute_convex_hull(verts)

        plot_word_hull(
            axes[i],
            verts,
            hull,
            COLORS[language],
            word,
            font
        )

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{language} Convex Hulls", fontsize=18)
    plt.tight_layout()
    plt.show()


def run_language_tests():
    """
    Main driver for testing all languages.
    Uses a small word list for quick visual checking.
    """
    test_words = {
        "English": ["hello", "world", "beautiful", "algorithm", "language",
                    "sky", "mountain", "river", "music", "dream"],
        "Chinese": ["你好", "世界", "美丽", "算法", "語言",
                    "天空", "山脉", "河流", "音樂", "梦想"],
        "Arabic":  ["مرحبا", "عالم", "جميل", "خوارزمية", "لغة",
                    "سماء", "جبل", "نهر", "موسيقى", "حلم"],
    }
    fonts = build_font_registry(list(test_words.keys()))
    for lang in test_words:
        visualize_language(lang, test_words[lang], fonts[lang])


#Write a function:
# Should return 10000 HULLS IN EACH LANGUAGE 
#STUCTURE: HULL POINTS, NUM of CHARACTERS, LANGUAGE LABEL, ORIGINAL WORD.
def generate_hull_dataset(words_by_lang: dict, font_registry: dict) -> list[dict]:
    """
    for every word compute its convex hull and store:
    - hull points
    - number of charaters
    - language
    - word
    """
    dataset = []
    for lang, words in words_by_lang.items():
        font = font_registry[lang]
        for word in words:
            record = process_word_to_record(word, lang, font)
            if record is not None:
                dataset.append(record)
    return dataset
    
def process_word_to_record(word: str, lang: str, font: FontProperties) -> dict | None:
    """
    convert a word into one dataset record.
    Returns None if the hull could not be computed.
    """
    verts = filter_anchor_points(render_text_path(word, font, size=100))
    hull  = compute_convex_hull(verts)
    if hull is None:
        return None
    return {
        "hull_points": verts[hull.vertices],  # just the boundary points
        "num_chars":   len(word),
        "language":    lang,
        "word":        word,
    }

# Step 1 — visual sanity check
run_language_tests()

# Step 2 — generate full dataset (uncomment when visuals look correct)
# full_words = load_word_lists(n_per_language=3333)
# font_registry = build_font_registry(list(full_words.keys()))
# dataset = generate_hull_dataset(full_words, font_registry)
# print(f"Generated {len(dataset)} records")