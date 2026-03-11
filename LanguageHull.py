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

FONT_PATHS = {
    "English": "fonts/NotoSans.ttf",
    "Chinese": "fonts/NotoSansTC.ttf",
    "Arabic":  "fonts/NotoSansArabic.ttf",
}

# easiest to test on a small set of words can expand one it works
WORDS = {
    "English": [
        "hello", "world", "beautiful", "algorithm", "language",
        "sky", "mountain", "river", "music", "dream",
    ],
    "Chinese": [
        "你好", "世界", "美丽", "算法", "语言",
        "天空", "山脉", "河流", "音乐", "梦想",
    ],
    "Arabic": [
        "مرحبا", "عالم", "جميل", "خوارزمية", "لغة",
        "سماء", "جبل", "نهر", "موسيقى", "حلم",
    ],
}

COLORS = {
    "English": "#4A90D9",
    "Chinese": "#E85D4A",
    "Arabic":  "#50C878",
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
    path = TextPath((0,0), word, seze=size, prop=font)
    return path.vertices