# Multi-Agent RAG System

A sophisticated Retrieval-Augmented Generation system built with LangGraph, featuring real-time feedback loops between multiple specialized agents for enhanced information retrieval and report generation.

## Project Overview

This project implements an advanced multi-agent RAG system designed for food industry knowledge management. The system uses intelligent query routing, parallel retrieval with inter-agent feedback, and comprehensive quality evaluation to deliver accurate, contextual responses.

## System Architecture

The system consists of 7 specialized agents working in coordinated workflows:

### Core Agents

**Planning Agent**
- Analyzes query complexity and determines processing strategy
- Routes simple queries to fast-track processing
- Decomposes complex queries into manageable sub-queries

**Retriever X Agent (Graph-focused)**
- Specializes in relationship exploration using Graph DB
- Sends real-time hints to Retriever Y based on discoveries
- Adapts search strategy based on feedback from other retrievers

**Retriever Y Agent (Multi-source)**
- Handles Vector DB, RDB, and Web search integration
- Processes hints from Retriever X to enhance search relevance
- Provides feedback for iterative search improvement

**Critic Agents (1 & 2)**
- Critic 1: Evaluates information sufficiency
- Critic 2: Assesses context quality and reliability
- Triggers additional search iterations when needed

**Context Integrator Agent**
- Synthesizes information from all retrieval sources
- Eliminates redundancy and resolves conflicting information
- Creates coherent, structured knowledge base

**Report Generator Agent**
- Produces final reports in multiple formats
- Supports PDF, mind map, and document generation
- Adapts output style based on query type

**Simple Answerer Agent**
- Handles straightforward queries efficiently
- Bypasses complex multi-agent pipeline for speed
- Uses lightweight Vector DB search

## Key Features

### Real-time Feedback Loop
- Asynchronous message passing between retrievers
- Dynamic search strategy adaptation
- Continuous quality improvement during processing

### Intelligent Query Classification
- Automatic complexity assessment
- Optimal resource allocation
- Fast-track processing for simple queries

### Multi-database Integration
- Graph DB for relationship mapping
- Vector DB for semantic search
- RDB for structured data queries
- Web search for real-time information

### Streaming Support
- Real-time progress monitoring
- Intermediate result streaming
- User experience optimization

### Quality Assurance
- Multi-stage evaluation process
- Information sufficiency validation
- Context quality assessment

## Technical Implementation

### Built With
- **LangGraph**: Multi-agent workflow orchestration
- **LangChain**: Agent framework and tool integration
- **OpenAI GPT**: Language model backend
- **Pydantic**: Data validation and modeling
- **Python AsyncIO**: Asynchronous processing

### Agent Communication
```python
class RealTimeFeedbackChannel:
    def __init__(self):
        self.x_to_y_queue = asyncio.Queue()
        self.y_to_x_queue = asyncio.Queue()
        self.active = asyncio.Event()
        self.message_history = []
```

### State Management
```python
class StreamingAgentState(BaseModel):
    original_query: str
    current_iteration: int = 0
    max_iterations: int = 2
    graph_results_stream: List[SearchResult] = []
    multi_source_results_stream: List[SearchResult] = []
    # ... additional state fields
```

## Usage Examples

### Basic Query Processing
```python
# Initialize workflow
workflow = RAGWorkflow()

# Run simple query
result = await workflow.run("What are the nutritional components of rice?")
print(result['final_answer'])
```

### Streaming Mode
```python
# Stream processing updates
async for step_result in workflow.stream_run("Analyze pea price trends and market forecast"):
    if step_result['status'] == 'processing':
        print(f"Step: {step_result['step']} - Progress updates")
    elif step_result['status'] == 'completed':
        print(f"Final Answer: {step_result['final_answer']}")
```

### Complex Analysis
```python
# Multi-source complex query
query = "Compare quinoa and oats nutritionally, include market trends and pricing analysis"
result = await workflow.run(query, max_iterations=3)

# Access detailed results
print(f"Complexity: {result['planning_result']['complexity']}")
print(f"Sources used: {result['search_results']['total_results']}")
print(f"Quality scores: Critic1={result['evaluations']['critic1_confidence']}")
```

## File Structure

```
multiagent-rag-system/
├── multi-agent-rag.ipynb          # Main implementation notebook
├── mock_databases.py              # Mock database implementations
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## Mock Database Structure

The system includes comprehensive mock databases for testing:

**Graph DB**: Node-relationship structures for ingredient connections
**Vector DB**: Document embeddings for semantic search
**RDB**: Structured price, nutrition, and market data
**Web Search**: Simulated news articles and trend data

## Key Improvements and Features

### Feedback-Enhanced Retrieval
- Dynamic search refinement based on inter-agent communication
- Reduced information gaps through collaborative discovery
- Improved relevance through iterative enhancement

### Quality Control Pipeline
- Multi-stage evaluation preventing insufficient responses
- Confidence scoring for reliability assessment
- Automatic retry mechanisms for quality assurance

### Performance Optimization
- Intelligent caching to prevent redundant operations
- Concurrent processing for speed improvement
- Resource-aware query routing

## Future Enhancements

- Integration with real production databases
- Advanced prompt engineering optimization
- Enhanced feedback loop mechanisms
- Performance benchmarking and validation
- Extended language model support

## Development Notes

This system demonstrates advanced RAG architecture patterns including:
- Multi-agent coordination and communication
- Real-time feedback integration
- Quality-driven iterative processing
- Scalable state management
- Comprehensive error handling

The implementation serves as a foundation for production-grade knowledge management systems requiring high accuracy and comprehensive information coverage.

## Author

Developed by Seongmin Lee (이성민)
Korea University, Computer Science & Engineering

## License

This project is available under the [MIT License](LICENSE).
