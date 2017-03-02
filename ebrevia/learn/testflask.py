from flask import Flask,request,redirect,url_for
import learner.Learner
import learner

app = Flask(__name__)

@app.route('/',methods=['POST'])
def classifyArff():
  print("Got arff")
  json = request.get_json()
  arff = json['arff']
  print(arff)
  return "Complete"
      
if __name__ == "__main__":
  prefix = "/Users/jakem/eb-files/pymodel/"
  provision='change_control'
  l = learner.load(prefix,provision)
  app.run()
