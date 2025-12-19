# <editor-fold desc="IMPORTS">
import sys
from txtai.embeddings import Embeddings
import sqlite3
import pickle
import heapq
# </editor-fold>

# <editor-fold desc="FILE PATHS">
model_path = r'C:\Users\visvesh\.cache\huggingface\hub\models--intfloat--e5-base\snapshots\b533fe4636f4a2507c08ddab40644d20b0006d6a'
embeddings_path = r'C:\Users\visvesh\.cache\txtai_embeddings'
filename = r'C:\Users\visvesh\Documents\instagram index.txt'
database = r'C:\Users\visvesh\Documents\store.db'
in_knn_path = r'C:\Users\visvesh\Documents\DBoperations\in_knn.pkl'
out_knn_path = r'C:\Users\visvesh\Documents\DBoperations\out_knn.pkl'
id_holes_path = r'C:\Users\visvesh\Documents\DBoperations\holes.pkl'
# filename = r'C:\Users\visvesh\Documents\youtube_index.txt';
# "C:\Users\visvesh\Documents\instagram index.txt"
# filename = r'C:\Users\visvesh\Documents\Instagram_Clusters.txt'
# Get-ExecutionPolicy -List in powershell for CurrentUser
# </editor-fold>

# <editor-fold desc="PARAMETERS">
K = 40
IN_KNN_CANDIDATE_LEN = 200
DELTA = 20
THRESHOLD = 7
IN_KNN_CANDIDATE_NEIGHBOUR_INDEX = int(K*0.75)
READ_BATCH_SIZE = 100
REFRESH_BATCH_SIZE = 100
TENTATIVE_KNN_FRACTION = 0.4
# </editor-fold>

# <editor-fold desc="LOAD DATA FROM STORAGE TO MEMORY">
out_knn = pickle.load(open(out_knn_path, "rb"))
in_knn = pickle.load(open(in_knn_path, "rb"))
holes = pickle.load(open(id_holes_path,"rb"))
embeddings = Embeddings()
embeddings.load(path=embeddings_path)
db = {}
conn = sqlite3.connect(database)
cursor = conn.cursor()
cursor.execute("select id, url, desc from main")
while True:
    rows = cursor.fetchmany(READ_BATCH_SIZE)
    if not rows:
        break
    batch_dict = {k: (v1, v2) for k, v1, v2 in rows} #url, description
    db.update(batch_dict)
refreshqueue = [set() for i in range(THRESHOLD+1)]
# </editor-fold>

# <editor-fold desc="NODE FUNCTIONS">
def in_knn_update(nodeid, old_knn, new_knn):
    # assumes id node is still present, it is job of caller to ensure this
    dicta = dict(old_knn[:K])
    dictb = dict(new_knn[:K])
    old_knn_only, new_knn_only = [dicta.keys() - dictb.keys()][0], [dictb.keys() - dicta.keys()][0]
    for i in old_knn_only:
        if i not in in_knn:
            continue
        idx = next(index for index,nid in enumerate(in_knn[i]) if nid == nodeid)
        del in_knn[i][idx]
    for i in new_knn_only:
        if i not in out_knn:
            continue
        in_knn[i].append(nodeid)

def refreshnode(nodeid):
    old_knn = out_knn[nodeid]
    results = embeddings.search(db[nodeid][1], K + DELTA + 1, "dense")
    new_knn = [(int(neighbour[0]), neighbour[1]) for neighbour in results[1:]]
    in_knn_update(nodeid, old_knn, new_knn)
    out_knn[nodeid] = new_knn

def insert(new_internet_data):
    new_internet_data_id = max(in_knn)+1 if len(holes)==0 else heapq.heappop(holes)
    candidatelistlen = IN_KNN_CANDIDATE_LEN
    tentative_in_knn = list()
    while True:
        tentative_in_knn.clear()
        bigquery = embeddings.search(new_internet_data, candidatelistlen, "dense", parameters={"efSearch": candidatelistlen})
        bigquery = dict(bigquery)
        for candidate_id in bigquery:
            if out_knn[candidate_id][IN_KNN_CANDIDATE_NEIGHBOUR_INDEX][1] < bigquery[candidate_id]:
                tentative_in_knn.append(candidate_id)
        if len(tentative_in_knn) < int(candidatelistlen * TENTATIVE_KNN_FRACTION):
            break
        else:
            candidatelistlen *= 2

    out_knn[new_internet_data_id] = list(bigquery.items())[:K+DELTA]
    in_knn[new_internet_data_id] = []
    in_knn_update(new_internet_data_id, [], out_knn[new_internet_data_id])
    for i in tentative_in_knn:
        old_knn = out_knn[i].copy()
        out_knn[i].append((new_internet_data_id, bigquery[i]))
        out_knn[i].sort(key=lambda x:x[1], reverse=True)
        out_knn[i] = out_knn[i][:K+DELTA]
        in_knn_update(i, old_knn, out_knn[i])
    embeddings.upsert([(new_internet_data_id, new_internet_data)])
    db[new_internet_data_id]=["URL_PLACEHOLDER", new_internet_data]

def delete(nodeid):
    # assumes node is present
    heapq.heappush(holes, nodeid)
    in_knn_update(nodeid, out_knn[nodeid], [])
    in_knn_id = in_knn[nodeid]
    del out_knn[nodeid]
    del in_knn[nodeid]
    embeddings.delete([nodeid])
    #need to rollback these maybe
    #db updatte also
    for i in in_knn_id:
        idx = next(j for j,(_,sim) in enumerate(out_knn[i]) if out_knn[i][j][0] == nodeid)
        del out_knn[i][idx]
        if out_knn[i][K-1][0] in in_knn:
            in_knn[out_knn[i][K-1][0]].append(i)
        #add to process underflow bucket
        knnbuffer = len(out_knn[i])-K
        if knnbuffer > THRESHOLD:
            continue
        elif knnbuffer == THRESHOLD:
            refreshqueue[THRESHOLD].add(i)
        elif knnbuffer in range(1, THRESHOLD):
            refreshqueue[knnbuffer].add(i)
            refreshqueue[knnbuffer+1].remove(i)
        else:
            refreshqueue[0].add(i)

def refreshbatch():
    ids = []
    for i in range(THRESHOLD+1):
        while len(ids)<REFRESH_BATCH_SIZE:
            if len(refreshqueue[i]) == 0:
                break
            nodeid = refreshqueue[i].pop()
            if nodeid in out_knn:
                ids.append(nodeid)
    if len(ids)==0:
        return
    #batch of size 100 is pretty optimal. 1.5s
    results = embeddings.batchsearch([db[nodeid][1] for nodeid in ids], K+DELTA+1, "dense")
    for i in range(len(ids)):
        new_knn = [(int(neighbour[0]), neighbour[1]) for neighbour in results[i][1:]]
        in_knn_update(ids[i], out_knn[ids[i]], new_knn)
        out_knn[ids[i]]=new_knn

def sanity(nodeid):
    for i,f in out_knn[nodeid][:K]:
        if i not in in_knn:
            return f"{nodeid} neighbour: {i} does not exist"
        elif nodeid not in in_knn[i]:
            return f"{nodeid} not in in_knn[{i}]"
    for i in in_knn[nodeid]:
        if i not in out_knn:
            return f"{nodeid} in {i}'s knn but {i} does not exist"
        elif nodeid not in list(zip(*out_knn[i][:40]))[0]:
            return f"{nodeid} not in out_knn[{i}]"
    for i in out_knn:
        if i not in in_knn:
            return f"out_knn[{i}], but no in_knn[{i}]"
    for i in in_knn:
        if i not in out_knn:
            return f"in_knn[{i}], but no out_knn[{i}]"
    return None
# </editor-fold>

# <editor-fold desc="I/O WITH JAVA SPRING BOOT (interactive)">
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
while True:
    task = input().split()
    if task[0] == "query":
        query_type = task[1]
        #input("input query to search for best matching URL (spelling mistakes result in mismatch)")
        result = []
        if query_type == "single":
            query_string = input()
            result = embeddings.search(query_string, K+DELTA, "dense") # [] or pair<string, float>
        bestmatches = []
        print(str(len(result)))
        for i,match in enumerate(result):
            print(str(match[0]) + " " + str(len(in_knn[match[0]])))
    else:
        print(str(3*len(db)))
        for key in db.keys():
            print(str(key)+'\n'+db[key][0]+'\n'+db[key][1])
# </editor-fold>

# import threading
# import time
#
# cs_lock = threading.Lock()
# fg_done = threading.Event()
# fg_done.set()  # initially, FG is "done"
#
# def foreground():
#     while True:
#         input()
#
#         fg_done.clear()  # FG wants next turn
#
#         with cs_lock:
#             # ---- critical section ----
#             print("foreground start")
#             time.sleep(3)
#             print("foreground done")
#
#         fg_done.set()  # FG done, BG can proceed
#
#
# def background():
#     while True:
#         # Wait until FG is done
#         fg_done.wait()  # blocks here until fg_done is set
#
#         with cs_lock:
#             # Re-check inside lock to avoid race
#             if not fg_done.is_set():
#                 continue
#
#             # ---- critical section ----
#             print("background")
#             time.sleep(1)

