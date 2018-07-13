from py2neo import Graph

graph = Graph(bolt=False, password='admin')

def WipeOut():
    graph.run('''
    MATCH (n)
    DETACH DELETE n
    ''')

def WriteRegulations(df):
    for i, row in df.iterrows():
        # print(row)
        title = row.titles
        reg_type = row.class_type
        filename = row.files
        isRepealed = bool(row.repealed)
        #     print(title,reg_type, filename, isRepealed)
        if isRepealed:
            graph.run('''
            MERGE (n: Regulation {Title:{T}, Classification:{C}})-[r:Repealed]->(f:File {Name:{N}})
            ''', {'T': title, 'C': reg_type, 'N': filename})
        else:
            graph.run('''
            MERGE (n: Regulation {Title:{T}, Classification:{C}})-[r:InForce]->(f:File {Name:{N}})
            ''', {'T': title, 'C': reg_type, 'N': filename})

def WriteActs(df):
    for i, row in df.iterrows():
        title = row.titles
        short_title = row.short_title
        isRepealed = bool(row.repealed)
        file = row.files
        if isRepealed:
            graph.run('''
            MERGE (n: Act {Title: {T}, ShortTitle: {S}})-[r:Repealed]->(f:File {Name: {N}})
            ''', {'T':title, 'S':short_title, 'N': file})
        else:
            graph.run('''
            MERGE (n: Act {Title: {T}, ShortTitle: {S}})-[r:InForce]->(f:File {Name: {N}})
            ''', {'T': title, 'S': short_title, 'N': file})