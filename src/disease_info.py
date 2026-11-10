"""
Disease Information Knowledge Base for KrishiDrishti.
Contains verified educational information, symptoms, preventive measures,
and basic agricultural recommendations for all PlantVillage tomato classes.
"""

from config.config import DEFAULT_TOMATO_CLASSES

DISEASE_KNOWLEDGE_BASE = {
    "Tomato___Bacterial_spot": {
        "display_name": "Tomato Bacterial Spot",
        "causal_agent": "Xanthomonas perforans / Xanthomonas vesicatoria (Bacterium)",
        "status": "Diseased",
        "severity": "High",
        "description": (
            "Bacterial spot is a widespread and potentially destructive disease of tomatoes. "
            "It affects all above-ground parts of the plant, causing fruit spots that reduce marketability "
            "and severe defoliation that exposes fruit to sunscald."
        ),
        "symptoms": [
            "Small, dark brown to black water-soaked circular spots on leaves (often under 3mm).",
            "Lesions may develop a faint yellow halo over time.",
            "Leaves turn yellow, dry up, and drop prematurely in severe infections.",
            "Rough, scabby, dark brown spots on green fruit.",
        ],
        "preventive_practices": [
            "Use certified pathogen-free seeds and disease-free transplants.",
            "Avoid overhead irrigation; use drip or furrow irrigation to keep foliage dry.",
            "Practice 2-3 year crop rotation away from solanaceous crops (potato, pepper, eggplant).",
            "Sanitize trellising stakes, pruning shears, and garden tools regularly.",
        ],
        "recommendations": [
            "Promptly prune and destroy heavily infected lower foliage.",
            "Apply approved copper-based organic protectant bactericides during wet, warm weather.",
            "Mulch around plant bases to prevent soil splashing onto bottom leaves.",
            "Ensure adequate spacing between plants to maximize air circulation.",
        ],
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato Early Blight",
        "causal_agent": "Alternaria solani (Fungus)",
        "status": "Diseased",
        "severity": "Moderate to High",
        "description": (
            "Early blight is a very common fungal disease affecting tomato crops worldwide. "
            "It typically begins on the oldest foliage near the ground and progresses upwards as the plant matures."
        ),
        "symptoms": [
            "Characteristic concentric ring pattern ('target board' or bullseye appearance) within brown spots.",
            "Yellowing tissue (chlorosis) surrounding the dark circular lesions.",
            "Stem lesions known as collar rot or dark sunken cankers.",
            "Dark, leathery sunken spots at the stem end of fruits.",
        ],
        "preventive_practices": [
            "Rotate crops with non-solanaceous plants on a 3-year schedule.",
            "Remove and compost or safely destroy old crop residues at season end.",
            "Maintain optimal plant nutrition; avoid nitrogen deficiency which increases susceptibility.",
            "Space plants adequately to facilitate quick drying of leaves after rain.",
        ],
        "recommendations": [
            "Stake or cage plants to elevate foliage off wet soil.",
            "Apply straw or plastic mulch around the root zone.",
            "Strip bottom leaves within 12-18 inches of the ground once plants are established.",
            "Apply bio-fungicides (e.g., Bacillus subtilis) or protective copper sprays at first sign.",
        ],
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato Late Blight",
        "causal_agent": "Phytophthora infestans (Oomycete / Water Mold)",
        "status": "Diseased",
        "severity": "Critical / High",
        "description": (
            "Late blight is a catastrophic, fast-spreading disease historically responsible for the Irish Potato Famine. "
            "Under cool, wet conditions, it can destroy entire tomato fields in just a few days if left unchecked."
        ),
        "symptoms": [
            "Large, irregular water-soaked dark greenish to brown lesions on leaves.",
            "Delicate white fungal-like mold or downy growth on leaf undersides in humid conditions.",
            "Dark brown, greasy lesions expanding rapidly across stems.",
            "Firm, greasy, olive-brown sunken patches on green or ripe fruits.",
        ],
        "preventive_practices": [
            "Select resistant cultivars whenever available.",
            "Eliminate volunteer tomato and potato plants in and around the field.",
            "Monitor local extension weather alerts for late blight forecasting.",
            "Keep foliage completely dry; avoid any overhead watering.",
        ],
        "recommendations": [
            "Immediately bag and remove severely infected plants to prevent airborne spore dissemination.",
            "Do not compost infected foliage in domestic compost heaps.",
            "Apply protective fungicides or bio-fungicides immediately upon disease detection in the region.",
            "Inform neighboring farms as late blight spores easily travel several kilometers in wind.",
        ],
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato Leaf Mold",
        "causal_agent": "Passalora fulva / Cladosporium fulvum (Fungus)",
        "status": "Diseased",
        "severity": "Moderate",
        "description": (
            "Leaf mold is primarily a foliage problem of tomatoes grown under high humidity, especially in greenhouses, "
            "high tunnels, and wet protected structures."
        ),
        "symptoms": [
            "Pale greenish-yellow chlorotic spots with indistinct margins on upper leaf surfaces.",
            "Dense olive-green to grayish-velvety fungal growth on the underside of corresponding spots.",
            "Infected leaves curl, wither, turn completely brown, and drop prematurely.",
            "Blossoms and fruit are rarely directly infected, but yield drops from defoliation.",
        ],
        "preventive_practices": [
            "Improve ventilation in polytunnels and greenhouses with circulation fans.",
            "Maintain relative humidity below 85% inside sheltered environments.",
            "Increase plant spacing to optimize airflow.",
            "Water only at ground level early in the morning.",
        ],
        "recommendations": [
            "Prune sucker branches and lower foliage to increase air movement.",
            "Sanitize greenhouse structures and tools thoroughly between crop cycles.",
            "Utilize resistant hybrid tomato varieties with Cf gene resistance.",
            "Apply preventive biological or sulfur-based sprays if humidity cannot be lowered.",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Tomato Septoria Leaf Spot",
        "causal_agent": "Septoria lycopersici (Fungus)",
        "status": "Diseased",
        "severity": "Moderate to High",
        "description": (
            "Septoria leaf spot is one of the most destructive foliage diseases of field tomatoes. "
            "While it rarely affects the fruit directly, heavy defoliation reduces fruit yield and induces sunscald."
        ),
        "symptoms": [
            "Numerous small (1.5 to 3 mm), circular spots with dark brown margins and grayish-white centers.",
            "Tiny black specks (pycnidia - spore-bearing fruiting bodies) visible in the pale centers under a magnifying glass.",
            "Progresses rapidly from bottom leaves upward.",
            "Leaves turn yellow, dry out, curl, and fall off.",
        ],
        "preventive_practices": [
            "Practice minimum 2-year crop rotation without solanaceous relatives.",
            "Deep plow or bury infected crop remnants after harvest.",
            "Control horsenettle and nightshade weeds which serve as alternative hosts.",
            "Apply thick organic mulch to block spore splash from soil.",
        ],
        "recommendations": [
            "Prune infected lower foliage as soon as tiny speckles appear.",
            "Avoid handling wet plants to prevent moving spores along rows.",
            "Water deeply at the base using soaker hoses or drip lines.",
            "Use copper-based or approved organic contact fungicides during persistent rainy spells.",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Tomato Spider Mite Infestation",
        "causal_agent": "Tetranychus urticae (Arachnid / Acari)",
        "status": "Diseased",
        "severity": "Moderate to High",
        "description": (
            "Two-spotted spider mites are microscopic arachnids that feed on plant sap by piercing leaf cells. "
            "Outbreaks are triggered by hot, dry, and dusty weather conditions."
        ),
        "symptoms": [
            "Fine yellow or white stippling / specks across upper leaf surfaces.",
            "Silken fine webbing visible on leaf undersides, growing tips, and leaf axils.",
            "Foliage becomes bronzed, bleached, dry, and brittle.",
            "Severe infestation causes complete leaf drop and stunted fruit growth.",
        ],
        "preventive_practices": [
            "Maintain adequate soil moisture; drought-stressed plants are preferred mite hosts.",
            "Wet unpaved farm lanes to reduce airborne dust that hinders predatory mites.",
            "Avoid excessive synthetic nitrogen fertilizers which boost mite reproduction.",
            "Preserve beneficial predatory insects (ladybugs, lacewings, predatory mites Phytoseiulus persimilis).",
        ],
        "recommendations": [
            "Spray underside of leaves with strong streams of water to dislodge mites and webs.",
            "Apply insecticidal soap, neem oil, or horticultural oils covering all foliage surfaces.",
            "Introduce commercially reared predatory mites in greenhouses.",
            "Avoid broad-spectrum synthetic insecticides which destroy natural predators.",
        ],
    },
    "Tomato___Target_Spot": {
        "display_name": "Tomato Target Spot",
        "causal_agent": "Corynespora cassiicola (Fungus)",
        "status": "Diseased",
        "severity": "Moderate to High",
        "description": (
            "Target spot is a fungal disease thriving in warm, humid tropical and subtropical climates. "
            "It affects leaves, stems, and fruits, causing substantial foliage damage and unmarketable fruit."
        ),
        "symptoms": [
            "Small water-soaked pinhead spots that enlarge to circular brown lesions.",
            "Lesions show concentric zones of alternating light and dark brown tissue, with yellow halos.",
            "Leaves turn yellow and drop prematurely, causing upper canopy thinning.",
            "Fruit lesions are sunken, circular, and crater-like with dark centers.",
        ],
        "preventive_practices": [
            "Select tolerant or resistant tomato cultivars.",
            "Maintain wide plant and row spacing to ensure thorough canopy ventilation.",
            "Avoid overhead sprinkler irrigation.",
            "Promptly eradicate solanaceous weed reservoirs.",
        ],
        "recommendations": [
            "Prune sucker branches and prune lower leaves to 40 cm above the ground.",
            "Apply balanced fertilizer; avoid excess nitrogen.",
            "Rotate crops with non-hosts such as corn or grasses.",
            "Spray preventative protective copper or bio-fungicides when weather turns consistently warm and humid.",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "causal_agent": "Begomovirus / Geminiviridae (Vector: Bemisia tabaci / Silverleaf Whitefly)",
        "status": "Diseased",
        "severity": "Critical / High",
        "description": (
            "TYLCV is one of the most economically devastating tomato viruses in warm regions. "
            "Transmitted persistently by tiny whiteflies, it severely stunts plant growth and halts fruit production."
        ),
        "symptoms": [
            "Upward curling, cupping, and crinkling of leaf margins ('spooning').",
            "Marked interveinal yellowing (chlorosis) on new young leaves.",
            "Severe stunting and dwarfing of plants with bushy, upright growth habits.",
            "Significant blossom drop; plants infected early bear virtually no fruit.",
        ],
        "preventive_practices": [
            "Plant certified TYLCV-resistant or tolerant hybrid varieties.",
            "Use physical 50-mesh insect-proof netting in nurseries and greenhouses.",
            "Eradicate infected plants immediately upon early visual diagnosis.",
            "Maintain a host-free crop break period between tomato growing seasons.",
        ],
        "recommendations": [
            "Install bright yellow sticky cards to monitor and trap adult whitefly populations.",
            "Spray organic neem seed extract or horticultural oil to smother whitefly nymphs.",
            "Introduce biological control agents like parasitic wasps (Encarsia formosa).",
            "Control alternative host weeds (such as nightshades, mallow, datura).",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato Mosaic Virus (ToMV)",
        "causal_agent": "Tobamovirus (Mechanically / Seed transmitted)",
        "status": "Diseased",
        "severity": "High",
        "description": (
            "Tomato mosaic virus is an exceptionally stable, contagious viral pathogen. "
            "Unlike insect-borne viruses, ToMV spreads primarily by mechanical contact, human hands, contaminated tools, "
            "and infected seeds, remaining viable in dry crop residue for years."
        ),
        "symptoms": [
            "Mottling of leaves with alternating light green and dark green mosaic patterns.",
            "Distorted, blistered, puckered, or narrow strap-like ('fern-leaf') leaves.",
            "Overall plant stunting and uneven ripening of fruits.",
            "Internal brown browning / necrotic patches inside harvested fruit flesh.",
        ],
        "preventive_practices": [
            "Plant virus-resistant tomato varieties (designated with 'T' or 'ToMV').",
            "Use certified heat-treated or trisodium phosphate (TSP) disinfected seed.",
            "Strictly wash hands with soap and water or 20% nonfat dry milk solution before working with plants.",
            "Prohibit tobacco smoking/chewing near tomatoes, as tobacco products can harbor the virus.",
        ],
        "recommendations": [
            "Disinfect all shears, stakes, and pruning tools in a 10% household bleach solution.",
            "Carefully rogue (dig up and discard) infected plants without brushing against neighboring healthy ones.",
            "Rotate crops for at least 2 seasons to clear virus-retaining root debris from the soil.",
            "Wash footwear and clothing when moving between different growing tunnels.",
        ],
    },
    "Tomato___healthy": {
        "display_name": "Healthy Tomato Leaf",
        "causal_agent": "None (Optimum Physiological Health)",
        "status": "Healthy",
        "severity": "None",
        "description": (
            "The leaf demonstrates healthy tissue without noticeable signs of bacterial, fungal, viral, or pest infections. "
            "Maintaining proactive cultural care is essential to sustain high yield and quality produce."
        ),
        "symptoms": [
            "Uniform vibrant green coloration across leaf lamina.",
            "Smooth or natural serrated margins without abnormal cupping or crinkling.",
            "Clear vein structure without chlorosis, mottling, or necrosis.",
            "Absence of spots, lesions, powdery coatings, or webbing.",
        ],
        "preventive_practices": [
            "Continue clean drip irrigation at root zones rather than wetting upper canopies.",
            "Maintain balanced soil fertility guided by regular soil tests (NPK + micronutrients).",
            "Keep base mulch in place to regulate soil temperature and suppress weeds.",
            "Scout plants twice weekly to detect any emerging pest or pathogen symptoms early.",
        ],
        "recommendations": [
            "Ensure regular trellising and gentle sucker removal to preserve light penetration.",
            "Apply preventive seaweed extract or beneficial mycorrhizae to boost immune vigor.",
            "Protect natural pollinator populations (bees) by avoiding indiscriminate spraying.",
            "Harvest fruit at peak physiological maturity to encourage continuous flowering.",
        ],
    },
}

# General agricultural disclaimer to show across the app
AGRICULTURAL_DISCLAIMER = (
    "Disclaimer: KrishiDrishti is an AI-assisted screening tool developed for academic and educational "
    "purposes under UN SDG 2 (Zero Hunger). Predictions are based on convolutional neural network image analysis "
    "and should not replace laboratory assays or on-field diagnosis by certified agronomists or plant pathologists. "
    "Always consult local agricultural extension services before administering chemical controls."
)


def get_disease_info(class_name: str) -> dict:
    """
    Retrieve disease information by class name.
    Falls back gracefully if an unmapped class is encountered.
    """
    if class_name in DISEASE_KNOWLEDGE_BASE:
        return DISEASE_KNOWLEDGE_BASE[class_name]

    # Flexible matching in case of slight naming variations
    for key, data in DISEASE_KNOWLEDGE_BASE.items():
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return data

    # Generic fallback
    clean_title = class_name.replace("Tomato___", "").replace("_", " ").title()
    is_healthy = "healthy" in class_name.lower()
    return {
        "display_name": f"Tomato: {clean_title}",
        "causal_agent": "Information pending verification",
        "status": "Healthy" if is_healthy else "Diseased",
        "severity": "None" if is_healthy else "Moderate",
        "description": f"Classification identified as {clean_title}. Ensure regular monitoring.",
        "symptoms": ["Consult local agricultural guide for full symptomatic profiling."],
        "preventive_practices": [
            "Ensure crop hygiene and sanitize farm equipment.",
            "Maintain proper plant spacing and drip irrigation.",
        ],
        "recommendations": [
            "Observe leaf progression over the next 48-72 hours.",
            "Consult your nearest Krishi Vigyan Kendra (KVK) or agricultural extension specialist.",
        ],
    }


def get_all_disease_keys() -> list:
    """Return all known class keys."""
    return list(DISEASE_KNOWLEDGE_BASE.keys())
