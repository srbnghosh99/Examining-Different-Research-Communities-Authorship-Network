import networkx as nx
import pandas as pd
from collections import Counter
import networkx as nx
from matplotlib import pylab as pl
import re
import matplotlib as plt
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



def year_by_calculation(file):
  nodelis = []
  edgelis = []
  yearlis = []
  cols = ['author1','author2', 'year']
  edges = pd.read_csv(file,names=cols ,delimiter = ',')
  print(edges)
  # print(G.number_of_nodes(), G.number_of_edges())

  # Create a graph for each year
  graphs_by_year = {}
  for index, row in edges.iterrows():
      # print(row)
      author1 = row['author1']
      author2 = row['author2']
      year = row['year']
      if year not in graphs_by_year:
          graphs_by_year[year] = nx.Graph()
      graphs_by_year[year].add_edge(author1, author2)

  # Now, calculate centrality for each year
  for year, graph in graphs_by_year.items():
      print(f"Year: {year}")
    #   print(graph.number_of_nodes(),graph.number_of_edges())
      yearlis.append(year)
      nodelis.append(graph.number_of_nodes())
      edgelis.append(graph.number_of_edges())
      print(str(year))
      filename =  str(year) +'.graphml'
      print(filename)
      nx.write_graphml(graph, filename)

'''
  df = pd.DataFrame({
    'Year': yearlis,
    'Nodes': nodelis,
    'Edges': edgelis
    })
  df = df.sort_values(by='Year').reset_index()
  df = df.drop(0)
  print(df)

  all = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/untitled folder/SW/all.csv')
  freq = all['Year'].value_counts().reset_index()
  print(freq)
  freq.columns = ['Year', 'No of papers']
  freq = freq.sort_values(by=['Year']).reset_index()
  freq = freq.drop(0)
  result = freq.merge(df,on='Year')
  result = result[['Year', 'Nodes', 'Edges', 'No of papers']]
  print(result)

  authors_by_year = list(graph.nodes())
  print(len(authors_by_year))
  degree_centrality = nx.degree_centrality(graph)
  print(f"Degree Centrality: {degree_centrality}")
  
  # Calculate communities (using a simple greedy modularity)
#   communities = nx.algorithms.community.greedy_modularity_communities(graph)
#   print(f"Communities: {list(communities)}")

  # Plot the network
  pl.figure()
  nx.draw(graph, with_labels=True)
  pl.title(f"Co-authorship Network for {year}")
  pl.show()
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--inputfilename",type = str)
    # parser.add_argument("--outputfilename",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.inputfilename)
    # print(inputs.outputfilename)
    # louvain_algo(inputs.inputfilename,inputs.outputfilename)
    year_by_calculation(inputs.inputfilename)
  

if __name__ == '__main__':
    main()
