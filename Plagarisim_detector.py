import ast
import re
import tokenize
from flask import Flask , request, render_template,jsonify
from io  import BytesIO
from difflib import SequenceMatcher


def preprocess(code):
    code = re.sub(r"#.*","",code)
    code = re.sub(r'\s+',' ',code).strip()
    return code



def tokenize_code(code):
    tokens = []

    tokens = [token.string for token in tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
              if token.type != tokenize.COMMENT and token.type != tokenize.NL and token.type != tokenize.ENCODING
             ]

    return tokens


def normalize_tokens(tokens):
    normal = []
    count = 0
    contents = {}
    
    for token in tokens:
        if token.isidentifier() and token not in ("def", "return", "class", "if", "else", "while"):
            if token not in contents:
                contents[token] = f"variable{count}"
                count += 1
            normal.append(contents[token])
        else:
            normal.append(token)
    
    return normal

def compare_tokens_with_sequenceMatcher(code1_tokens,code2_tokens):
    ret = SequenceMatcher(None,code1_tokens,code2_tokens)        
    return ret.ratio() * 100







def parse_code_to_ast(code):
    try:
        tree = ast.parse(code)
        return tree
    except SyntaxError:
        return None

class ASTFeatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.features = []

    def generic_visit(self, node):
        self.features.append(type(node).__name__)
        super().generic_visit(node)

    def visit_FunctionDef(self, node):
        self.features.append(f"Function: {node.name}")
        super().generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.features.append(f"Call: {node.func.id}")
        super().generic_visit(node)

def compare_ast_features(features1, features2):
    matcher = SequenceMatcher(None, features1, features2)
    return matcher.ratio() * 100

def detect_ast_similarity(code1, code2):
    tree1 = parse_code_to_ast(code1)
    tree2 = parse_code_to_ast(code2)
    
    if tree1 is None or tree2 is None:
        return 0
    
    extractor1 = ASTFeatureExtractor()
    extractor2 = ASTFeatureExtractor()
    
    extractor1.visit(tree1)
    extractor2.visit(tree2)
    
    features1 = extractor1.features
    features2 = extractor2.features
    
    return compare_ast_features(features1, features2)






def tokens_techinque(input1,input2):
    preprocess1 = preprocess(input1)
    preprocess2 = preprocess(input2)
    tokens1 = tokenize_code(preprocess1)
    tokens2 = tokenize_code(preprocess2)
    normal1 = normalize_tokens(tokens1)
    normal2 = normalize_tokens(tokens2)
    comparison_result = compare_tokens_with_sequenceMatcher(normal1,normal2)
    
    return comparison_result
    
def classify_similarity(token_similarity, ast_similarity):
    if token_similarity >= 70 and ast_similarity >= 65:
        return "Plagiarized"
    elif token_similarity >= 70 and ast_similarity > 50:
        return "Possibly Plagiarized"
    elif token_similarity < 70 and ast_similarity >= 50:
        return "Similar Logic"
    elif token_similarity >= 70 and ast_similarity >= 40:
        return "Different Logic"
    else:
        return "Not Plagiarized"






app = Flask(__name__)


@app.route('/')
def home():
    return render_template("Main.html")

@app.route("/checkSimilarity",methods = ['post'])
def checkSimilarity():
    
    data = request.json
    code1 = data.get("code1","")
    code2 = data.get("code2","")
    
    tokens_result = tokens_techinque(code1,code2)  
    ast_result = detect_ast_similarity(code1,code2) 
    classification = classify_similarity(tokens_result,ast_result)
    return jsonify({"token_similarity" : tokens_result,
                    "ast_similarity" : ast_result,
                   "classification": classification
                   })
        
if __name__ == '__main__':
    app.run(debug=True)
