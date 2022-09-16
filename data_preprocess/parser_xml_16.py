import xml.sax
import json
 
class MovieHandler(xml.sax.ContentHandler):
   def __init__(self):
      self.CurrentData = ""
      self.text = ""
      self.correction = ""
      self.errors = []
      self.parser_data = []
    #   self.error = ''
      self.id = ''
      # self.rating = ""
      # self.stars = ""
      # self.description = ""
 
   # 元素开始事件处理
   def startElement(self, tag, attributes):
      self.CurrentData = tag
      if tag == "DOC":
         print("*****DOC*****")
      if tag == 'ERROR':
          self.errors.append({'start': int(attributes["start_off"])-1, 'end': int(attributes["end_off"]), 'type': attributes["type"]})
      if tag == 'TEXT':
          self.id = attributes['id']
         #  if self.id == '13114200304131645200064_2_6x2':
         #     print(1)
         # title = attributes["text"]
         # print "Title:", title
 
   # 元素结束事件处理
   def endElement(self, tag):
      if self.CurrentData == "TEXT":
         print("ID:", self.id)
         print("TEXT:", self.text)
      elif self.CurrentData == "CORRECTION":
         print("CORRECTION:", self.correction)
      elif self.CurrentData == "ERROR":
         print("ERROR:", self.errors)
      elif tag == 'DOC':
          self.parser_data.append({'id': self.id, 'text': self.text, 'correct': self.correction, 'error': self.errors})
          self.errors = []
          self.text = ''
          self.correction = ''
      # elif self.CurrentData == "rating":
      #    print "Rating:", self.rating
      # elif self.CurrentData == "stars":
      #    print "Stars:", self.stars
      # elif self.CurrentData == "description":
      #    print "Description:", self.description
      self.CurrentData = ""
 
   # 内容事件处理
   def characters(self, content):
      if self.CurrentData == "TEXT":
         self.text += content
      elif self.CurrentData == "CORRECTION":
         self.correction += content
    #   elif self.CurrentData == "ERROR":
    #      self.error = content
      # elif self.CurrentData == "rating":
      #    self.rating = content
      # elif self.CurrentData == "stars":
      #    self.stars = content
      # elif self.CurrentData == "description":
      #    self.description = content
  
if __name__ == "__main__":
   
   # 创建一个 XMLReader
   parser = xml.sax.make_parser()
   # turn off namepsaces
   parser.setFeature(xml.sax.handler.feature_namespaces, 0)
 
   # 重写 ContextHandler
   Handler = MovieHandler()
   parser.setContentHandler(Handler)
   
   parser.parse("/home/liyunliang/CGED_Task/cged_datasets/cged2016/nlptea16cged_release1.0/Training/CGED16_HSK_TrainingSet_m.xml")
   # parser.parse("/home/liyunliang/CGED_Task/dataset/cged2017/CGED17_HSK_TrainingSet_m.xml")
   # parser.parse("/home/liyunliang/CGED_Task/dataset/cged2018/CGED18_HSK_TrainingSet_m.xml")
   # parser.parse('/home/liyunliang/CGED_Task/dataset/test18.xml')

   # with open('/home/liyunliang/CGED_Task/dataset/cged2018/CGED18_HSK_TrainingSet_m.jsonl', 'w') as f:
   with open('/home/liyunliang/CGED_Task/cged_datasets/cged2016/nlptea16cged_release1.0/Training/CGED16_HSK_TrainingSet_m.jsonl', 'w') as f:
   # with open('/home/liyunliang/CGED_Task/dataset/cged2017/CGED17_HSK_TrainingSet_m.jsonl', 'w') as f:
      f.write('\n'.join([json.dumps(item, ensure_ascii=False) for item in Handler.parser_data]))