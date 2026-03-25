import sys
sys.path.append('c:/Users/radhe/Desktop/toxlens-ai-main/backend')
from model.predict import predict_smiles
import pprint

# Ethanol and TNT
print("Running test prediction with SHAP...")
res = predict_smiles("CCO", "C1=CC=C(C=C1)[N+](=O)[O-]")
pprint.pprint(res)
