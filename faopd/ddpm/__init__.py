# Import DDPM modules conditionally
try:
    from .ddpm_customize import MLPDiffusionCustomize
except ImportError as e:
    print(f"Warning: MLPDiffusionCustomize import failed: {e}")
    MLPDiffusionCustomize = None

try:
    from .ddpm_condition import MLPConditionDiffusion
except ImportError as e:
    print(f"Warning: MLPConditionDiffusion import failed: {e}")
    MLPConditionDiffusion = None
