# FP Visualization
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.DataManip.Metric import GetTanimotoDistMat
from rdkit.DataManip.Metric import GetTanimotoSimMat
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import IPythonConsole




f = open(r'data/chemical.smiles','r')
smiles = []; names = []
for line in f.readlines()[1:]:
	itm = line.strip().split('\t')
	smiles = itm[0].strip()
	name = itm[1].strip()

	ms = Chem.MolFromSmiles(smiles)
	Draw.MolToFile(ms,r'FP_visualization\%s.png'%name)  # draw molecule
	
	FP = [727,531]
	bi = {}
	fp = AllChem.GetMorganFingerprintAsBitVect(ms, radius=2, nBits=1024, bitInfo=bi)

	# img = Draw.DrawMorganBit(ms, 33, bi)
	# tpls = [(ms,x,bi) for x in FP]
	# img2 = Draw.DrawMorganBits(tpls,molsPerRow=6,legends=['FP_'+str(i) for i in FP], returnPNG=False) 
	keys = bi.keys()
	for fbnumber in FP:
		if fbnumber-1 in keys:
			img = Draw.DrawMorganBit(ms, int(fbnumber-1), bi)
			img.save(r'FP_visualization\%s_FP_%s.png'%(name,str(fbnumber)),dpi=(1000,1000)) # Draw FP
			# svg = img.GetDrawingText()
			# with open(r'FP_visualization\%s_FP_%s.svg'%(name,str(fbnumber)), "w") as file:
			# 	file.write(img)

	# img2.save(r'FP_visualization\%s_FP.png'%(name),optimize=True, quality=5000) # Draw FP
f.close()

