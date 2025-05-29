import datetime
from flask import Flask
from flask import jsonify
import operator
import re

maximum_distance = 100

# Read the dataset and store in a dictionary #
with open('word_search.tsv') as file:
    rows = (line.rstrip("\n").split('\t') for line in file)
    d = {row[0]: row[1] for row in rows}

#Initializing the Flask app #
app = Flask(__name__)

# Search API #
@app.route('/search/<name>', methods=['GET'])
def index(name):
    start = datetime.datetime.utcnow()
    op_list=[]
    word_list=[]
    lis_op=[]
    dictList=[]
    new_list_val = []
    reg_match = allmatch(name,d.keys())
    for i in reg_match:
        op=levenshtein(name,i)
        op_list.append(op)
        word_list.append(i)

    dictList.append(dict(zip(word_list, op_list)))
    op_list=sorted(dictList[0].items(), key=operator.itemgetter(1))

    for list_val in op_list:
        new_list_val.append(list_val[0])

    print(datetime.datetime.utcnow() - start)
    return jsonify({"result":new_list_val[:25]})

# Regex matching #
def allmatch(name, all_data):
    top_most = []
    pattern = '.*?'.join(name)
    regex = re.compile(pattern)
    for item in all_data:
        match = regex.search(item)
        if match:
            top_most.append((len(match.group()), match.start(), item))
    return [x for _, _, x in sorted(top_most)]

# Calculating the minimum edit distance #
def levenshtein(seq1, seq2):
    first = None
    row = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        second, first, row = first, row, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            del_cost = first[y] + 1
            add_cost = row[y - 1] + 1
            sub_cost = first[y - 1] + (seq1[x] != seq2[y])
            row[y] = min(del_cost, add_cost, sub_cost)
    return row[len(seq2) - 1]

if __name__ == '__main__':
    app.run(debug=False)