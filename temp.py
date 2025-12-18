import sys
from contextlib import nullcontext

import txtai

#Get-ExecutionPolicy -List in powershell for CurrentUser

# model_path = r'C:\Users\visvesh\.cache\huggingface\hub\models--sentence-transformers--nli-mpnet-base-v2\snapshots\c2f4dd9a1dc4337c28cfbd650433f761bb304c50'
model_path = r'C:\Users\visvesh\.cache\huggingface\hub\models--intfloat--e5-small-v2\snapshots\ffb93f3bd4047442299a41ebb6fa998a38507c52';

sys.stdout.reconfigure(encoding='utf-8')
embeddings = txtai.Embeddings()
if len(sys.argv) > 1:
    embeddings.load(path=r'C:\Users\visvesh\.cache\txtai_embeddings')
else:
    # Create an embeddings (small 87MB huggingface all-MiniLM-L6-v2/nli-mpnet-base-v2 for token matching)
    embeddings = txtai.Embeddings(
        path=model_path,
        index={
            "type": "ivf",
            "nlist": 100}
    );

    # "C:\Users\visvesh\Documents\instagram index.txt"
    # filename = r'C:\Users\visvesh\Documents\Instagram_Clusters.txt'
    filename = r'C:\Users\visvesh\Documents\instagram index.txt';
    data = []

    # open file and read line by line
    with open(filename, "r", encoding='utf-8') as f:
        for line in f:
            data.append(" ".join(line.rstrip('\n').split(" ")[1:]))
            print(line.rstrip('\n'))

    # Create an index for the list of text and embed into ~740 dimension vector space
    embeddings.index(data)
    embeddings.save(r'C:\Users\visvesh\.cache\txtai_embeddings')
    exit(0)

query_string = sys.argv[1]
#input("input query to search for best matching URL (spelling mistakes result in mismatch)")
result = embeddings.search(query_string, 40, "dense") # [] or pair<string, float>

bestmatches = []

for match in result:
    print(str(match[0]) + " " + str(match[1]))