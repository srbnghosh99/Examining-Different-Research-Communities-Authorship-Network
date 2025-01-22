import networkx as nx
import pandas as pd
from collections import Counter
import networkx as nx
from matplotlib import pylab as pl
import re
import matplotlib as plt
from networkx.readwrite import json_graph
from itertools import permutations,combinations
import json


# df1 = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/untitled folder/SW/ICASE.csv')
# df2 = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/untitled folder/SW/ICSE.csv')
# df3 = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/untitled folder/SW/SIGSOFT.csv')
# df = pd.concat([df1,df2,df3])
# df.to_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/untitled folder/SW/all.csv')
# df = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/untitled folder/SW/all.csv')
# df.shape

df1 = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/scholardata/dm/ICDM.csv')
df2 = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/scholardata/dm/ICMLA.csv')
df3 = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/scholardata/dm/SIGKDD.csv')
df = pd.concat([df1,df2,df3])
df.to_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/dm/all.csv')
df = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/dm/all.csv')
df.shape


def preprocess():

  df['Year'] = ""
  for index, row in df.iterrows():
    info = (row['publication_info'])
    year = re.findall(r'\d+', info)
    if year:
      df.at[index,'Year'] = year[-1]

  df['formatted_authors_name'] = ""
  for index, row in df.iterrows():
    lis = (row['authors_name']).split(',')
    # print(lis)
    if lis:
      l = []
      for i in lis:
        i = i.strip()
        i = i.split(' ')
        j = "_".join(i)
        # print(j)
        # j = str(j).replace('[','').replace(']','').replace('\"','')
        j = str(j).replace('[','').replace(']','')
        # print(j)
        l.append((j))
      df.at[index,'formatted_authors_name'] = l


  df['formatted_authors_id'] = ""
  for index, row in df.iterrows():
    lis = (row['authors_id']).split(',')
    if (lis):
      l = []
      for i in lis:
        j = i
        j = str(j).replace('[','').replace(']','')
        l.append(str(j))
      df.at[index,'formatted_authors_id'] = l

def id_name_mapping():
  import ast
  df = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/scholardata/sw/all.csv')
  df_new = df[['authors_id','formatted_authors_name']]
  df_new['formatted_authors_name'] = df_new['formatted_authors_name'].apply(ast.literal_eval)
  df_new['authors_id'] = df_new['authors_id'].apply(ast.literal_eval)
  id_name_mapping = {}

  # Iterate through each row of the DataFrame
  for index, row in df_new.iterrows():
      ids = [id.strip().replace("'", "") for id in row['authors_id']]
      names = [name.strip().replace("'", "") for name in row['formatted_authors_name']]
      
      # Iterate through the IDs and names in each row
      for id, name in zip(ids, names):
          if id and id not in id_name_mapping:  # Check if ID is not already in the dictionary and is not empty
              id_name_mapping[id] = name

  df_mapping = pd.DataFrame.from_dict(id_name_mapping, orient='index', columns=['formatted_authors_name']).reset_index()

  # Rename the columns to reflect the original structure
  df_mapping.columns = ['authors_id', 'formatted_authors_name']

  # Display the new DataFrame
  df_mapping.to_csv('sw/author_id_name.csv', index = False)
  # Group by 'formatted_authors_name' and count unique IDs
  duplicate_names = df_mapping.groupby('authors_id')['formatted_authors_name'].nunique().reset_index()
  print(duplicate_names.sort_values(by='formatted_authors_name', ascending=False))

  duplicate_names = df_mapping.groupby('formatted_authors_name')['authors_id'].nunique().reset_index()
  print(duplicate_names.sort_values(by='authors_id', ascending=False))



def create_graph():

  import ast
  df = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/scholardata/sw/all.csv')
  df_mapping = pd.read_csv('sw/author_id_name.csv')
  G = nx.Graph()
  for index, row in df_mapping.iterrows():
      G.add_node(row['authors_id'], name=row['formatted_authors_name'])

  df['authors_id'] = df['authors_id'].apply(ast.literal_eval)
  for index, row in df.iterrows():
      if len(row['authors_id']) > 1:
          lis = row['authors_id']
          list_combinations = list(combinations(lis, 2))
          # print(list_combinations)
          for edge in list_combinations:
              # print(edge[0],edge[1])
              G.add_edge(edge[0], edge[1])
          # break
  print(G.number_of_nodes(), G.number_of_edges())
  nx.number_of_selfloops(G)
  self_loop_nodes = [node for node in G.nodes() if G.has_edge(node, node)]
  print("Nodes with self-loops:", self_loop_nodes)

  print("Nodes in the graph:", G.nodes(data=True))
  nx.write_graphml(G, "sw/coauthor_net_id_sw.graphml")
  nx.write_graphml(G, "sw/coauthor_net_id_sw.adjlist")

  degree_data = G.degree()

  # Convert to a DataFrame
  degree_df = pd.DataFrame(degree_data, columns=['Node', 'Degree'])


  # Assuming df is your DataFrame
  lis = df['formatted_authors_name'].tolist()
  yearlis = df['Year'].tolist()
  count = 0
  with open('coauthor_net_named_sw.txt', 'w') as f:
      for i in range(len(lis)):
          # print(len(lis[i]))
          # Generate combinations
          if len(lis[i]) == 1:
              # print(lis[i])
              name = lis[i][-1]
              # print(name)
              list_combinations = [(name, name)]
              # print(list_combinations)
              count += 1
              # break
          else:
              list_combinations = list(combinations(lis[i], 2))
              # print(list_combinations)
          
          # Write to file
          for comb in list_combinations:
              # print(comb)
              if '' not in comb:
                      # Format the combinatio n as a space-separated string
                  strippedText = ' '.join(comb).strip()
                  # print(strippedText)
                  f.write(f"{strippedText} {yearlis[i]}\n")
  print('Number of self loops or single author',count)
               
G = nx.read_weighted_edgelist('coauthor_net_named_sw.txt')
print(G.number_of_nodes(), G.number_of_edges())

def affiliation_dataframe():
  aff = pd.read_csv('dm/data_mining_large_cluster_author_affiliations.csv')
  aff['cited_by'] = aff['cited_by'].apply(ast.literal_eval)
  def clean_cited_by(value):
      if isinstance(value, list) and len(value) > 0:
          # Extract the first element and try to convert to integer, handle 'None' as 0
          try:
              return int(value[0])
          except ValueError:
              return 0  # Handle cases where 'None' or invalid string exists
      return 0

  # Apply the function to the 'cited_by' column
  aff['cited_by'] = aff['cited_by'].apply(clean_cited_by)
  aff = aff[['author_name', 'author_id', 'email',
        'cited_by', 'name', 'affiliations', 'split_author_name', 'name1']]
  aff.to_csv('dm/data_mining_large_cluster_author_affiliations.csv')

def influential_author():
  cols = ['A1','A2', 'year']
  df = pd.read_csv('coauthor_net_named_sw.txt', names=cols, delimiter = ' ')
  df1 = df['A1']
  df2 = df['A2']
  frames = [df1, df2]
  result = pd.concat(frames)
  df3 = pd.DataFrame(result)
  author_freq = df3[0].value_counts().rename_axis('unique_values').reset_index(name='counts')
  author_freq = author_freq.sort_values(by=['counts'],ascending=False)
  top_10 = author_freq.head(10)
  top_10



def affiliation():
  df = pd.read_csv('/Users/shrabanighosh/My work/UNCC/Fall 2024/untitled folder/SW/software_engg_large_cluster_author_affiliations_2 (1).csv',sep = ',')
  df["filtered_email"] = ""
  for index, row in df.iterrows():
    text = (row['email'])
    emails = re.findall(r"[a-z0-9\.\-+_]+\.[a-z0-9\.\-+_]+\.[a-z]+", text)
    if emails == []:
      emails = re.findall(r"[a-z0-9\.\-+_]+\.[a-z]+", text)
    df.at[index,'filtered_email'] = emails

  higher = df['filtered_email'].value_counts()
  higher.head(30)  


def year_by_calculation():
  cols = ['author1','author2', 'year']
  edges = pd.read_csv('coauthor_net_named_sw.txt',names=cols ,delimiter = ' ')
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
      degree_centrality = nx.degree_centrality(graph)
      print(f"Degree Centrality: {degree_centrality}")
      
      # Calculate communities (using a simple greedy modularity)
      communities = nx.algorithms.community.greedy_modularity_communities(graph)
      print(f"Communities: {list(communities)}")

      # Plot the network
      plt.figure()
      nx.draw(graph, with_labels=True)
      plt.title(f"Co-authorship Network for {year}")
      plt.show()


def graph_to_json():
  G = nx.read_edgelist("SW/coauthor_net_named_sw.txt", nodetype=str, data=(("weight", int),))
  print(G.number_of_nodes(), G.number_of_edges())
  with open('SW/coauthor_net_named_sw.json', 'w') as outfile1:
    outfile1.write(json.dumps(json_graph.node_link_data(G)))
  # python3 /Users/shrabanighosh/My work/UNCC/Summer 2024/itsc-2214-readings-main/Community-Detection-in-Large-Graphs-and-Applications/TEST/Untitled/Use_Cases/Trust_Prediction/create_node_propensity.py --dataset SW --inDirectory single_community --outDirectory propensity_single_community


def centrality_measure():
  import networkx as nx
  G = nx.read_edgelist("SW/coauthor_net_named_sw.txt", nodetype=str, data = False)
  print(G.number_of_nodes(), G.number_of_edges())
  deg_centrality = nx.degree_centrality(G)
  bet_centrality = nx.betweenness_centrality(G, normalized = True, endpoints = False) 
  pr = nx.pagerank(G, alpha = 0.8) 
  close_centrality = nx.closeness_centrality(G) 
  
  G = nx.read_edgelist('SW/coauthor_net_named_sw.txt', create_using=nx.DiGraph(), nodetype=str, data = False)
  out_deg_centrality = nx.out_degree_centrality(G)   
  in_deg_centrality = nx.in_degree_centrality(G) 
  
  

  deg_centrality =  pd.DataFrame.from_dict(deg_centrality, orient='index')
  deg_centrality  =deg_centrality.reset_index()
  deg_centrality = deg_centrality.rename(columns={'index': 'author', 0: 'value'})
  deg_centrality = deg_centrality.sort_values(by = 'value',ascending= False)
  deg_centrality.to_csv('SW/in_deg_centrality.csv', index = False)


def top_influencer_centrality():
  import os
  from os.path import dirname, join as pjoin
  curr_directory = os.getcwd()
  directory = 'propensity_single_community'
  path = pjoin(curr_directory,directory)
  csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]
  for index, file in enumerate(csv_files):
    # Read the CSV file
    df = pd.read_csv(os.path.join(path, file))
    top = df.head(5)
    print(top)



def largest_cc():
  G = nx.read_edgelist("SW/coauthor_net_named_sw.txt", nodetype=str, data=(("weight", int),))
  print(G.number_of_nodes(), G.number_of_edges())
  largest_cc = max(nx.connected_components(G), key=len)
  S = [G.subgraph(c).copy() for c in nx.connected_components(G)]


def largest_cc_authors_centrality():
  cols = ['A1','A2', '{}']
  df = pd.read_csv('SW/largest_sw.csv', names=cols, delimiter = ' ')
  df = df.drop('{}', axis=1)
  df1 = df['A1']
  df2 = df['A2']
  frames = [df1, df2]
  result = pd.concat(frames)
  df3 = pd.DataFrame(result)
  df3 = df3[0].drop_duplicates()
  df3 = df3.reset_index()
  df3 = df3.rename(columns={0: 'author'})
  df4 = pd.read_csv('SW/propensity_single_community/close_centrality.csv')
  common_authors = df3.merge(df4, on = 'author')
  # common_authors[common_authors['author'] == "'R_Buyya'"]
  common_authors = common_authors[['author', 'value']]
  # common_authors
  centrality = common_authors.sort_values(by='value', ascending=False)
  centrality['nodesize'] = centrality.apply(lambda row: row.value*1000, axis=1)
  centrality.nodesize = centrality.nodesize.round(0)
  centrality['nodesize']=centrality['nodesize'].astype(int)
  # betweenness_centrality.head(20)
  centrality
  G = nx.read_edgelist("SW/largest_sw.csv", nodetype=str, data=False)
  print(G.number_of_nodes(), G.number_of_edges())
  for index, row in centrality.iterrows():
      node = row['author']
      G.nodes[node]['centrality'] = row['nodesize']
  for u,outer_d in G.nodes(data=True):
      print(u,outer_d)
  nx.write_graphml(G, 'largest_sw_with_closeness.graphml')



def co_authored_frequency_weight():
  cols = ['A1','A2', '{}']
  df = pd.read_csv('SW/coauthor_net_named_sw.txt', names= cols, delimiter = ' ')
  df = df.drop('{}', axis=1)
  df1 = df['A1']
  df2 = df['A2']
  frames = [df1, df2]
  result = pd.concat(frames)
  df4 = pd.DataFrame(result)
  author_id = df4[0].unique().tolist()
  lis = list(range(1,len(author_id)))
  len(author_id),len(lis)
  res = res = dict(zip(author_id, lis))
  df['A3'] = df['A1'].map(res)
  df['A4'] = df['A2'].map(res)
  df['A4'] = df['A4'].fillna(0)
  df['A4'] = df['A4'].round().astype('Int64')
  freq = df.groupby(["A3", "A4"]).size().reset_index(name="Weight")
  freq = freq.sort_values(by=['Weight'],ascending=False).reset_index()

def density():
  G = nx.read_edgelist("SW/coauthor_net_named_sw.txt", nodetype=str, data = False)
  print(G.number_of_nodes(), G.number_of_edges())
  self_loops = [(node, node) for node in G.nodes() if G.has_edge(node, node)]
  G.remove_edges_from(self_loops)
  nx.density(G)
  G = nx.read_edgelist("dm/coauthor_net_named_dm.txt", nodetype=str, data = False)
  print(G.number_of_nodes(), G.number_of_edges())
  self_loops = [(node, node) for node in G.nodes() if G.has_edge(node, node)]
  G.remove_edges_from(self_loops)
  nx.density(G)