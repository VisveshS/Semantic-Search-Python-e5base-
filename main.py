# <editor-fold desc="IMPORTS">
import sys
from csv import excel

from pandas.core.strings.accessor import cat_core
from txtai.embeddings import Embeddings
import sqlite3
import pickle
import heapq
import threading
import time
# </editor-fold>

# <editor-fold desc="FILE PATHS">
model_path = r'C:\Users\visvesh\.cache\huggingface\hub\models--intfloat--e5-base\snapshots\b533fe4636f4a2507c08ddab40644d20b0006d6a'
embeddings_path = r'C:\Users\visvesh\.cache\txtai_embeddings'
filename = r'C:\Users\visvesh\Documents\instagram index.txt'
database = r'C:\Users\visvesh\Documents\store.db'
in_knns_path = r'C:\Users\visvesh\Documents\DBoperations\in_knn.pkl'
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
REFRESH_QUEUE_OVERLOAD_LIMIT = 500
TENTATIVE_KNN_FRACTION = 0.4
EVENTTIMEOUT = 3 #seconds
LONGWAIT_PAGERANK = 15 #seconds
# </editor-fold>

# <editor-fold desc="DATA STRUCTURES (NEEDS DSLOCK TO ACCESS)">
DSlock = threading.Lock() # data structures lock
fg_idle = threading.Event()
small_knn_detected = threading.Event()
save_to_disk = threading.Event(); save_to_disk.set()
out_knn = pickle.load(open(out_knn_path, "rb"))
in_knns = pickle.load(open(in_knns_path, "rb"))
in_knn = in_knns["in_knn"]
in_knn_buffer = in_knns["in_knn_buffer"]
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
conn.close()
refreshqueue = [set() for i in range(THRESHOLD+1)]
refreshqueue_overload = False
useractive = True
pagerank = None
pagerank_stale = 20
# </editor-fold>

# <editor-fold desc="NODE FUNCTIONS">
def in_knn_update(nodeid, in_knn_arr, in_knn_type, old_knn, new_knn):
    # assumes id node is still present, it is job of caller to ensure this
    dicta = dict(old_knn[:K]) if in_knn_type == "active" else dict(old_knn[K:])
    dictb = dict(new_knn[:K]) if in_knn_type == "active" else dict(new_knn[K:])
    old_knn_only, new_knn_only = [dicta.keys() - dictb.keys()][0], [dictb.keys() - dicta.keys()][0]
    for i in old_knn_only:
        if i not in in_knn_arr:
            continue
        idx = in_knn_arr[i].index(nodeid)
        del in_knn_arr[i][idx]
    for i in new_knn_only:
        if i not in in_knn_arr:
            continue
        in_knn_arr[i].append(nodeid)

def refreshnode(nodeid):
    if nodeid not in out_knn:
        return 0,False
    old_knn = out_knn[nodeid]
    results = embeddings.search(db[nodeid][1], K + DELTA + 1, "dense")
    new_knn = [(int(neighbour[0]), neighbour[1]) for neighbour in results[1:]]
    in_knn_update(nodeid, in_knn, "active",old_knn, new_knn)
    in_knn_update(nodeid, in_knn_buffer, "buffer",old_knn, new_knn)
    out_knn[nodeid] = new_knn
    return clustering_score([nodeid]+[i for i,j in out_knn[nodeid][:K]]), True

def insert(new_internet_url, new_internet_data):
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

    # duplicate URL not allowed, but duplicate description ok
    for i in bigquery:
        if db[i][0] == new_internet_url:
            return clustering_score([i]+[i1 for i1,j in out_knn[i][:K]]),i
    global pagerank_stale
    pagerank_stale = 20
    pagerank[new_internet_data_id] = 0.0
    out_knn[new_internet_data_id] = list(bigquery.items())[:K+DELTA]
    in_knn[new_internet_data_id] = []
    in_knn_buffer[new_internet_data_id] = []
    in_knn_update(new_internet_data_id, in_knn, "active", [], out_knn[new_internet_data_id])
    in_knn_update(new_internet_data_id, in_knn_buffer, "buffer", [], out_knn[new_internet_data_id])
    for i in tentative_in_knn:
        old_knn = out_knn[i].copy()
        out_knn[i].append((new_internet_data_id, bigquery[i]))
        out_knn[i].sort(key=lambda x:x[1], reverse=True)
        out_knn[i] = out_knn[i][:K+DELTA]
        in_knn_update(i, in_knn, "active",old_knn, out_knn[i])
        in_knn_update(i, in_knn_buffer, "buffer",old_knn, out_knn[i])
    embeddings.upsert([(new_internet_data_id, new_internet_data)])
    db[new_internet_data_id]=[new_internet_url, new_internet_data]
    return clustering_score([new_internet_data_id]+[i for i,j in out_knn[new_internet_data_id][:K]]), new_internet_data_id

def delete(nodeid):
    if nodeid not in out_knn:
        return 0
    heapq.heappush(holes, nodeid)
    in_knn_update(nodeid, in_knn, "active",out_knn[nodeid], [])
    in_knn_update(nodeid, in_knn_buffer, "buffer",out_knn[nodeid], [])
    in_knn_id = in_knn[nodeid]+in_knn_buffer[nodeid]
    cscore = clustering_score([i for i,j in out_knn[nodeid][:K]])
    del pagerank[nodeid]
    del out_knn[nodeid]
    del in_knn[nodeid]
    del in_knn_buffer[nodeid]
    embeddings.delete([nodeid])
    for i in in_knn_id:
        old_knn = out_knn[i].copy()
        idx = [nid for nid,sim in out_knn[i]].index(nodeid)
        del out_knn[i][idx]
        in_knn_update(i, in_knn, "active",old_knn, out_knn[i])
        in_knn_update(i, in_knn_buffer, "buffer",old_knn, out_knn[i])
        knnbuffer = len(out_knn[i])-K
        if knnbuffer > THRESHOLD:
            continue
        if knnbuffer == THRESHOLD:
            refreshqueue[THRESHOLD].add(i)
        elif knnbuffer in range(1, THRESHOLD):
            refreshqueue[knnbuffer].add(i)
            refreshqueue[knnbuffer+1].remove(i)
        else:
            refreshqueue[0].add(i)
    global pagerank_stale
    pagerank_stale = 20
    if any(refreshqueue):
        small_knn_detected.set()
        save_to_disk.clear()
    return cscore

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
        in_knn_update(ids[i], in_knn, "active", out_knn[ids[i]], new_knn)
        in_knn_update(ids[i], in_knn_buffer, "buffer", out_knn[ids[i]], new_knn)
        out_knn[ids[i]]=new_knn

def consistency(nodeid):
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

def mutual_knn(nodeid):
    return "".join(['1' if i in in_knn[nodeid] else '0' for i,j in out_knn[nodeid]])

def clustering_score(nodes):
    nodes = set(nodes)
    num_neighbours = 0
    for nodeid in nodes:
        in_knn_set = set(in_knn[nodeid])
        num_neighbours += len(in_knn_set & nodes)
    return num_neighbours/(len(nodes)*(len(nodes)+1))
# </editor-fold>

# <editor-fold desc="BACKGROUND">
def pagerank_from_in_knn(in_knn_arr, pr, num_iters=1, d=0.85):
    #100K operations per second, min 1 iteration, max 20
    N = len(in_knn_arr)
    out_degree = 40  # given
    allkeys = list(in_knn.keys())
    if pr is None:
        pr = dict.fromkeys(allkeys, 1.0 / N)
    for _ in range(num_iters):
        new_pr = dict.fromkeys(allkeys, (1 - d) / N)
        for i in allkeys:
            for j in in_knn[i]:
                new_pr[i] += d * pr[j] / out_degree
        pr = new_pr
    return pr
def pagerank_refresh():
    global pagerank, pagerank_stale
    longwait = LONGWAIT_PAGERANK
    while useractive:
        time.sleep(0.05)
        if pagerank_stale == 0 and longwait != 0: #not stritly thread safe, but not a critical operation, so its tolerable
            longwait -= 1
            time.sleep(1)
        else:
            fg_idle.wait(timeout=EVENTTIMEOUT) #wait until fg is done with processing user activity
            with DSlock:
                if not fg_idle.is_set():
                    continue
                print("pagerank compute triggered")
                pagerank = pagerank_from_in_knn(in_knn, pagerank)
                pagerank_stale = max(pagerank_stale-1,0)
                if pagerank_stale == 0:
                    longwait = LONGWAIT_PAGERANK
def background_batch_refresh():
    while useractive:
        small_knn_detected.wait(timeout=EVENTTIMEOUT) # wait until nodes with small knn lengths are detected, if drop below K+threshold
        fg_idle.wait(timeout=EVENTTIMEOUT) #wait until fg is done with processing user activity
        with DSlock:
            if not fg_idle.is_set():
                continue
            refreshbatch()
            if not any(refreshqueue):
                small_knn_detected.clear()
                save_to_disk.set()
        if sum(len(i) for i in refreshqueue) >= REFRESH_QUEUE_OVERLOAD_LIMIT:
            time.sleep(1)
knn_refresh_thread = threading.Thread(target=background_batch_refresh)
pagerank_refresh_thread = threading.Thread(target=pagerank_refresh)
# </editor-fold>

# <editor-fold desc="I/O WITH JAVA SPRING BOOT (interactive)">
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
while useractive:
    task = input()
    match task:
        case "setup":
            fg_idle.clear()
            with DSlock:
                knn_refresh_thread.start()
                pagerank_refresh_thread.start()
            fg_idle.set()
            print(len(db))
        case "querynode":
            nodeid = int(input())
            fg_idle.clear()
            with DSlock:
                cluster_score, nodeexist = refreshnode(nodeid) #user may request deleted node (clicking back and clicking prev deleted row)
                print('1' if nodeexist else '0')
                if not nodeexist:
                    continue
                nodeid_knn = [i for i, j in out_knn[nodeid][:40]]
                nodeid_in_knn_count = len(in_knn[nodeid])
                pr = 0 if pagerank is None else pagerank[nodeid] * len(pagerank)
                in_knn_counts = [len(in_knn[i]) for i in nodeid_knn]
                prs = [0 if pagerank is None else pagerank[i] * len(pagerank) for i in nodeid_knn]
                cansave = '1' if save_to_disk.is_set() else '0'
                mutual_knns = mutual_knn(nodeid)
            fg_idle.set()
            print(db[nodeid][0])
            print(db[nodeid][1])
            print(nodeid_in_knn_count)
            print(pr)
            print(mutual_knns)
            print(cluster_score)
            print(cansave)
            print(len(nodeid_knn))
            for i, neighbour in enumerate(nodeid_knn):
                print(db[neighbour][0])
                print(db[neighbour][1])
                print(neighbour, in_knn_counts[i], prs[i])
        case "querystring":
            textquery = input()
            results = embeddings.search(textquery, K, "dense")
            mutual_knns=""
            fg_idle.clear()
            with DSlock:
                in_knn_counts = [len(in_knn[i]) for i,j in results]
                for candidate_neighbour, score in results:
                    mutual_knns += '1' if out_knn[candidate_neighbour][IN_KNN_CANDIDATE_NEIGHBOUR_INDEX][1] < score else '0'
                results = [i for i, j in results]
                cscore = clustering_score(results)
                prs = [0 if pagerank is None else pagerank[i] * len(pagerank) for i in results]
                cansave = '1' if save_to_disk.is_set() else '0'
            fg_idle.set()
            print(cscore)
            print(mutual_knns)
            print(cansave)
            print(len(results))
            for i, neighbour in enumerate(results):
                print(db[neighbour][0])
                print(db[neighbour][1])
                print(neighbour, in_knn_counts[i], prs[i])
        case "insert":
            textquery = input().split(" ", 1)
            fg_idle.clear()
            with DSlock:
                cscore, new_index = insert(textquery[0], textquery[1])
                cansave = '1' if save_to_disk.is_set() else '0'
                len_in_knn = len(in_knn[new_index])
                pr = 0 if pagerank is None else pagerank[new_index] * len(pagerank)
                mutual_knns = mutual_knn(new_index)
            fg_idle.set()
            print(f"{new_index}\n{len_in_knn}\n{pr}\n{mutual_knns}\n{cscore}\n{cansave}")
        case "delete":
            nodeid = int(input())
            fg_idle.clear()
            with DSlock:
                cscore = delete(nodeid)
            fg_idle.set()
            print(f"{cscore}\ndeleted")
        case "consistency": #for debugging only. not used in deployment
            nodeid = int(input())
            consistent_check = consistency(nodeid)
            print("CONSISTENT" if consistent_check is not None else consistent_check)
        case "save":
            while True:
                save_to_disk.wait(EVENTTIMEOUT)
                with DSlock:
                    if not save_to_disk.is_set():
                        continue
                    postsave = input()
                    pickle.dump(out_knn, open(out_knn_path, "wb"))
                    pickle.dump({"in_knn":in_knn, "in_knn_buffer":in_knn_buffer}, open(in_knns_path, "wb"))
                    pickle.dump(holes, open(id_holes_path, "wb"))
                    embeddings.save(r'C:\Users\visvesh\.cache\txtai_embeddings')
                    conn = sqlite3.connect(database)
                    cursor = conn.cursor()
                    for id in db:
                        cursor.execute('''
                            INSERT INTO main (id, url, desc)
                            VALUES (?, ?, ?)
                            ON CONFLICT(id) DO UPDATE SET
                                url = excluded.url,
                                desc = excluded.desc
                        ''', (id, db[id][0], db[id][1]))
                    conn.commit()
                    conn.close()
                    if postsave == "browse":
                        print("saved")
                        break
                    else:
                        useractive = False
                        break
        case "exit":
            useractive = False
# except(ex):
#
#     useractive = False
# </editor-fold>
