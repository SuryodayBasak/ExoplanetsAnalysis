import requests
import sys

url = 'http://www.hpcf.upr.edu/~abel/phl/phl_hec_all_confirmed.csv.zip'  
print("Attempting to download dataset.")
r = requests.get(url)

try:
	with open("dataset.zip", "wb") as code:
		code.write(r.content)
	print("Data retrieval successful.")

except:
	print("Error in downloading file!")
	sys.exit(0)
	

