from langgraph.graph import StateGraph, START, END

from analyze_papers import (
    MyState,
    user_input,
    search_open_alex,
    search_arxiv,
    read_papers
)

from research_agent import (
    research_agent,
    decision,
    research_gaps,
    analyze_papers
)


graph = StateGraph(MyState)


graph.add_node('user_input', user_input)

graph.add_node('research_agent', research_agent)

graph.add_node('search_open_alex', search_open_alex)

graph.add_node('search_arxiv', search_arxiv)

graph.add_node('read_papers', read_papers)

graph.add_node('analyze_papers', analyze_papers)

graph.add_node('research_gaps', research_gaps)


graph.add_edge(START, 'user_input')

graph.add_edge('user_input', 'research_agent')


graph.add_conditional_edges(
    'research_agent',
    decision,
    {
        'arxiv': 'search_arxiv',
        'alex': 'search_open_alex',
        'read': 'read_papers',
        'finish': END
    }
)


graph.add_edge('search_open_alex', 'research_agent')

graph.add_edge('search_arxiv', 'research_agent')


graph.add_edge('read_papers', 'analyze_papers')

graph.add_edge('analyze_papers', 'research_gaps')

graph.add_edge('research_gaps', END)


workflow = graph.compile()

result = workflow.invoke({})