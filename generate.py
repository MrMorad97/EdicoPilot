from graphviz import Digraph

def generate_rag_diagram(filename="rag_system_diagram"):
    dot = Digraph(comment="RAG System Architecture", format='png')
    dot.attr(rankdir='TB', size='8,6')
    dot.attr('node', shape='box', style='rounded,filled', color='lightblue2', fontname='Helvetica')

    # Nodes
    dot.node('A', 'User Query\n(Input Question)')
    dot.node('B', 'PDF Ingestion & Processing\n- Load PDFs\n- Clean & Split Text\n- Create Embeddings')
    dot.node('C', 'Vector Store & Retriever\n(Chroma DB)')
    dot.node('D', 'Retrieve Top-K Relevant Docs')
    dot.node('E', 'Context Formatter\n(Concatenate & Trim)')
    dot.node('F', 'Prompt Builder\n(Insert Context + Question)')
    dot.node('G', 'Streaming Language Model\n(ChatOllama LLM)')
    dot.node('H', 'Streaming Output\n(Real-time Answer)')
    dot.node('I', 'Response Cache\n(Reused Answers)')

    # Edges
    dot.edge('A', 'C', label='Query')
    dot.edge('B', 'C', label='Indexed Docs & Embeddings')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    dot.edge('F', 'G')
    dot.edge('G', 'H')
    dot.edge('F', 'I', style='dashed', label='Check Cache')
    dot.edge('I', 'H', style='dashed', label='Cached Response')

    # Optional: Indicate direct chat mode (no documents)
    dot.node('J', 'Direct Chat Mode\n(No PDFs Ingested)', shape='ellipse', style='filled', color='lightgray')
    dot.edge('A', 'J', style='dotted', label='If no docs')

    dot.render(filename, view=True)

if __name__ == "__main__":
    generate_rag_diagram()
