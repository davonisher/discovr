import re

def classify_price_and_quality(brand: str, price: float) -> dict:
    """
    Bepaal op basis van brand & price de prijscategorie en kwaliteitscategorie.
    Hier gebruiken we statische dicts, maar je kunt het uitbreiden
    met data uit je eigen database of een LLM.
    """
    brand_tiers = {
        "domyos":       {"price_tier": "budget", "quality_tier": "low_quality"},
        "technogym":    {"price_tier": "premium", "quality_tier": "high_quality"},
        "adidas":       {"price_tier": "mid_range", "quality_tier": "medium_quality"},
        "nike":         {"price_tier": "mid_range", "quality_tier": "medium_quality"},
        "hammer":       {"price_tier": "budget", "quality_tier": "medium_quality"},
    }
    brand_clean = re.sub(r'\W+', '', brand.lower())

    if brand_clean in brand_tiers:
        price_cat = brand_tiers[brand_clean]["price_tier"]
        quality_cat = brand_tiers[brand_clean]["quality_tier"]
    else:
        price_cat = determine_price_category(price)
        quality_cat = "unknown"

    # Voorbeeld: is de prijs > 500, noem het altijd premium
    if price > 500:
        price_cat = "premium"

    return {
        "price_category": price_cat,
        "quality_category": quality_cat
    }

def determine_price_category(price: float) -> str:
    """
    Simple voorbeeldverdeling, breid naar wens uit.
    """
    if price < 20:
        return "very_cheap"
    elif price < 50:
        return "budget"
    elif price < 100:
        return "mid_range"
    elif price < 300:
        return "high_end"
    else:
        return "premium"
