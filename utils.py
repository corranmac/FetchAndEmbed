import pandas as pd
from xml.etree import ElementTree as ET
from sentence_transformers import CrossEncoder
from IPython.display import clear_output
import numpy as np

import sys,os
sys.path.append("/content/vila/examples/end2end-sci-pdf-parsing")
sys.path.append("/content/vila/src/vila")

import layoutparser as lp # For visualization 
from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import HierarchicalPDFPredictor, LayoutIndicatorPDFPredictor

from main import pipeline
from pathlib import Path
import io
import pandas as pd
import time

from sklearn.metrics.pairwise import cosine_similarity

def getLayout(pdfpath,pdf_extractor,vision_model1,vision_model2,pdf_predictor, page_tokens, page_images):

  layout_csv = pipeline(input_pdf=Path(f"{pdfpath}"),
              return_csv=True,
              pdf_extractor=pdf_extractor,
              vision_model1=vision_model1,
              vision_model2=vision_model2,
              pdf_predictor=pdf_predictor,
              relative_coordinates=False,
              page_tokens=page_tokens,
              page_images=page_images,
          )

  parsed=io.StringIO(layout_csv.to_csv(index=False))
  parsed = pd.read_csv(parsed)
  parsed= parsed.round({'x1': 1, 'x2': 0, 'y1': 1, 'x1': 1}) # Rounds values to reduce sensitivity

  # Sorts paragraphs into reading order by page, section, then by x1 and finally y1
  sectionNo=0
  for i, row in parsed.iterrows():
      if row['type'] == 'Section':
        sectionNo+=1
      parsed.loc[i, "id"] = sectionNo
  parsed.sort_values(by=["page","id", "x1", "y1"], ascending=[True, True, True,True], inplace=True)
  return parsed, pdfpath

def getPDFTree(parsed, pdfpath):
  pdf_metadata={}
  pdf_sections=[]

  root = ET.Element('root')
  article = ET.Element('article')
  title = ET.SubElement(article, 'title')
  authors = ET.SubElement(article, 'authors')
  abstract = ET.SubElement(article, 'abstract')
  keywords = ET.SubElement(article, 'keywords')
  sections = ET.SubElement(article, 'sections')
  section = None
  textL = []
  sectionL= []

  for i, row in parsed.iterrows():
    if row['type']=='Title':
      title.text = row['text']
    if row['type']=='Author':
      authors.text = row['text']
    if row['type']=='Abstract':
      abstract.text = row['text'].replace("Abstract:","").replace("Abstract","")
    if row['type']=='Keywords':
      keywords.text = row['text'].replace("Keywords:","").replace("Keywords","")

    if row['type']=='Section':
      section = ET.SubElement(sections, "Section")
      section.set("title", row['text'])
    elif row['type']=='Paragraph':
      if section == None:
        section = ET.SubElement(sections, "Section")
      paragraph = ET.SubElement(section, 'paragraph')
      paragraph.text = row['text']
      sectionL.extend([row['type']])
      textL.extend([row['text']])

  #root.append(article)
  #tree = ET.ElementTree(root)
  #tree.write('article.xml', xml_declaration=True, encoding='utf-8')
  paragraphs = {'id':ids,'page':pages,'text':textItems,'x1':x1s,'x2':x2s,'y1':y1s,'y2':y2s}
  data = {'path': pdfpath, 'sections': sectionL, 'chunk': textL}
  df = pd.DataFrame.from_dict(data)

  return textL, df

def getDF(parsed, pdfpath):
  meta={}
  paragraphs=[]
  section = ""

  for i, row in parsed.iterrows():
    if row['type']=='Title':
      meta['title'] = row['text']
    if row['type']=='Author':
      meta['authors'] = row['text']
    if row['type']=='Abstract':
      meta['abstract'] = row['text'].replace("Abstract:","").replace("Abstract","")
    if row['type']=='Keywords':
      meta['keywords'] = row['text'].replace("Keywords:","").replace("Keywords","")
    if row['type']=='Section':
      section = row['text']
    elif row['type']=='Paragraph':
      if section == None:
        continue
      paragraphs.append({'id':str(i),'page':row['page'],'section':section,'text':row['text'],'x1':row['x1'],'x2':row['x2'],'y1':row['y1'],'y2':row['y2']})

  return meta,paragraphs


def getRetriever(model_name, device):
  try:
    if 'instructor' in model_name:
      from InstructorEmbedding import INSTRUCTOR
      model = INSTRUCTOR(model_name, device=device)
    elif 'sentence-transformers' in model_name:
      from sentence_transformers import SentenceTransformer
      model = SentenceTransformer(model_name)
  except:
    return "Invalid model"
  return model

def getReranker(model_name, device):
  try:
    if 'cross-encoder' in model_name:
      from sentence_transformers import CrossEncoder
      model = CrossEncoder(model_name, device=device)
  except:
    return "Invalid model"
  return model
  
def embedPassages(passages,model,instruction,batch_size):
  if not instruction == None:
    passages = [[instruction,passage] for passage in passages]
  batches =  [passages[i : i + batch_size] for i in range(0, len(passages), batch_size)]
  embeddings = []
  for batch in batches:
    embeddings.extend(model.embed_text(batch))
  return embeddings

def getTokenLength(text,model): # requires tokenizer
  e = model.tokenizer(text)
  return {'n_chars': len(text),'tokens': len(e['input_ids'])}


def simplyRetrieve(query,qInstruction,data,model,n_results): #Gets top n results, requires sentence transformer
  query_embedding = [retriever.embed_text(query)]
  similarities = cosine_similarity(query_embedding,data['embeddings'])[0]

  sim_scores_argsort = reversed(np.argsort(similarities))
  resultData = {'reScore':[]}
  for key in list(data.keys()):
    resultData[key] = []

  c=0
  for doc in sim_scores_argsort:
    if c<n_results:
      for key in list(data.keys()):
        resultData[key].append(data[key][doc])
      resultData['reScore'].append(similarities[doc])
      c+=1
  return resultData

def reRank(data, reranker, query, n_results): #Requires ranker CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
  model_inputs = [[query, passage] for passage in data['text']]
  scores = reranker.predict(model_inputs)

  #Sort the scores in decreasing order
  #results = [{'input': inp, 'score': score} for inp, score in zip(model_inputs, scores)]
  sim_scores_argsort = reversed(np.argsort(scores))

  resultData = {'raScore':[]}
  for key in list(data.keys()):
    resultData[key] = []
  c=0
  for doc in sim_scores_argsort:
    if c<n_results:
      for key in list(data.keys()):
        resultData[key].append(data[key][doc])
      resultData['raScore'].append(scores[doc])
      c+=1
  return resultData
