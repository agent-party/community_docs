# Technical Architecture

## System Overview: The Clubhouse

Agent Party's core system, **Clubhouse**, is a graph-based collaboration framework that enables AI agents to evolve, communicate, and collaborate through a standardized protocol. The name "Clubhouse" reflects our vision: a central gathering place where AI capabilities come together, interact, and create emergent intelligence.

## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Primary Language** | Python 3.10+ | Main implementation language |
| **Web Framework** | FastAPI | API and WebSocket endpoints |
| **Graph Database** | Neo4j | Entity storage and relationship mapping |
| **Event Stream** | Kafka | Reliable event processing |
| **Cache/State** | Redis | In-memory data and session management |
| **Object Storage** | MinIO | File and artifact management |
| **ML Framework** | PyTorch Geometric | GNN implementation |
| **Container Runtime** | Docker | Component isolation and deployment |
| **Orchestration** | Kubernetes | Scaling and service management |
| **Vector Search** | Neo4j Vector Search | Similarity-based knowledge retrieval |
| **OAuth Provider** | Auth0/Keycloak | External system authentication |

### Development Tools

| Tool | Purpose |
|------|---------|
| **Pydantic** | Data validation and schema definition |
| **Protocol Classes** | Service interface contracts |
| **pytest** | Testing framework with coverage metrics |
| **mypy** | Static type checking |
| **black & isort** | Code formatting |
| **Hypothesis** | Property-based testing |
| **Docker Compose** | Local development environment |

## Architecture Patterns

### Service Registry Pattern

All services in Clubhouse are registered through a centralized Service Registry, which:
- Provides dependency injection for components
- Manages service lifecycle and initialization
- Enables mocking and testing of service dependencies
- Controls service scoping (singleton, transient, scoped)

```python
class ServiceRegistry:
    """Central registry for all system services."""
    
    def __init__(self):
        self._services = {}
        
    def register(self, service_type: Type, implementation: Any, scope: str = "singleton"):
        """Register a service implementation for a given interface type."""
        self._services[service_type] = {
            "implementation": implementation,
            "scope": scope,
            "instance": None if scope != "singleton" else implementation
        }
        
    def get_service(self, service_type: Type) -> Any:
        """Retrieve a service of the specified type."""
        if service_type not in self._services:
            raise ServiceNotFound(f"Service {service_type.__name__} not registered")
            
        service = self._services[service_type]
        
        if service["scope"] == "singleton":
            return service["instance"]
        elif service["scope"] == "transient":
            return service["implementation"]()
        # Handle other scopes as needed
```

### Repository Pattern

Data access is implemented through repositories, which:
- Abstract database operations from business logic
- Provide strongly-typed access methods
- Implement proper transaction management
- Optimize Neo4j queries with appropriate indexing

```python
class AgentRepository(Protocol):
    """Interface for agent data access."""
    
    async def get_agent_by_id(self, agent_id: str) -> Agent:
        """Retrieve an agent by ID."""
        ...
        
    async def create_agent(self, agent: Agent) -> str:
        """Create a new agent record."""
        ...
        
    async def update_agent_state(self, agent_id: str, state: str) -> None:
        """Update an agent's state."""
        ...
```

### Event-Driven Architecture

The system uses event-driven communication with:
- Strongly-typed event definitions using Pydantic
- Reliable event delivery through Kafka
- Idempotent event handlers
- Clear event ownership boundaries

```python
class AgentStateChangedEvent(BaseModel):
    """Event triggered when an agent changes state."""
    
    agent_id: str
    previous_state: str
    new_state: str
    timestamp: datetime
    transition_approver: str | None
    transition_reason: str
    metrics: dict[str, Any]
```

## Component Architecture

### Service Layer Components

```
┌───────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                       │
└───────────────▲───────────────────────────────▲───────────────┘
                │                               │
                │                               │
┌───────────────▼───────────┐   ┌───────────────▼───────────────┐
│  Service Layer            │   │  WebSocket Manager            │
│                           │   │                               │
│  - AgentService           │   │  - ConnectionManager          │
│  - TeamService            │   │  - MessageHandler             │
│  - TaskService            │   │  - VisualizationPublisher     │
│  - TemplateService        │   │  - EventSubscriber            │
└───────────────▲───────────┘   └───────────────────────────────┘
                │
                │
┌───────────────▼───────────────────────────────────────────────┐
│  Domain Services                                              │
│                                                               │
│  - AgentLifecycleManager                                      │
│  - TeamAssemblyService                                        │
│  - GNNRecommendationService                                   │
│  - TaskAnalysisService                                        │
└───────────────▲───────────────────────────────▲───────────────┘
                │                               │
                │                               │
┌───────────────▼───────────┐   ┌───────────────▼───────────────┐
│  Repository Layer         │   │  Infrastructure Services      │
│                           │   │                               │
│  - AgentRepository        │   │  - KafkaEventPublisher        │
│  - TeamRepository         │   │  - MinIOStorageService        │
│  - TaskRepository         │   │  - RedisStateManager          │
│  - TemplateRepository     │   │  - ModelProviderService       │
│  - WorkflowRepository     │   │  - OAuthIntegrationService    │
└───────────────▲───────────┘   └───────────────▲───────────────┘
                │                               │
                │                               │
┌───────────────▼───────────┐   ┌───────────────▼───────────────┐
│  Neo4j Database           │   │  External Services            │
└───────────────────────────┘   └───────────────────────────────┘
```

### Agent System Components

```
┌───────────────────────────┐
│  Template Registry         │
│                           │
│  - Template management    │
│  - Capability registry    │
│  - Version control        │
└───────────────▲───────────┘
                │
                │
┌───────────────▼───────────┐   ┌───────────────────────────────┐
│  Agent Factory            │   │  Talent Scout Agent           │
│                           │   │                               │
│  - Agent instantiation    │◄──┤  - Template creation          │
│  - Parameter validation   │   │  - Capability analysis        │
│  - Resource allocation    │   │  - Template optimization      │
└───────────────▲───────────┘   └───────────────────────────────┘
                │
                │
┌───────────────▼───────────┐
│  Lifecycle Manager        │
│                           │
│  - State transitions      │
│  - Approval workflows     │
│  - Event generation       │
│  - Resource tracking      │
└───────────────▲───────────┘
                │
                │
┌───────────────▼───────────┐   ┌───────────────────────────────┐
│  Agent Runtime            │   │  Manager Agents               │
│                           │   │                               │
│  - Execution environment  │◄──┤  - Transition approval        │
│  - Context management     │   │  - Policy enforcement         │
│  - Output handling        │   │  - Performance monitoring     │
│  - Tool integration       │   │  - Cost accounting            │
└───────────────────────────┘   └───────────────────────────────┘
```

## The Graph Neural Network (GNN) Architecture

Clubhouse uses a Graph Neural Network (GNN) as its core intelligent orchestration system. The GNN powers several critical capabilities:

### Agent Evolution and Cold Start Solution

One of the fundamental challenges in multi-agent systems is the "cold start problem": how do new agents efficiently integrate into existing workflows without extensive training or configuration? 

Our GNN-based approach addresses this through:

1. **Capability Inheritance**: New agents can inherit capabilities from similar agents in the graph
2. **Collaboration Templates**: Pre-established patterns of successful collaborations are encoded as templates
3. **Dynamic Knowledge Transfer**: Context, skills, and information flow dynamically between related agents
4. **Experience Embedding**: Agent experiences are encoded in the graph structure and node embeddings

```python
class GNNOrchestrator:
    """
    Evolutionary orchestrator that recommends optimal agent collaborations.
    Functions as the "meta-agent" or "orchestrator" that evolves the agent ecosystem.
    """
    
    def __init__(
        self, 
        graph_service: GraphService, 
        model_service: ModelProviderService, 
        config: GNNConfig
    ):
        self.graph_service = graph_service
        self.model_service = model_service
        self.config = config
        self.gnn_model = self._initialize_gnn_model()
        
    @log_execution_time
    async def find_optimal_team(self, task: Task, constraints: TeamConstraints) -> Team:
        """
        Find the optimal team of agents for a given task, solving the cold-start problem
        through contextual pattern matching and capability analysis.
        """
        # Extract task requirements
        requirements = await self._extract_task_requirements(task)
        
        # Get capabilities graph with agent nodes
        graph = await self.graph_service.get_capability_subgraph(
            requirements=requirements,
            max_depth=self.config.max_graph_depth
        )
        
        # Apply GNN model to predict optimal team composition
        team_composition = self._apply_gnn_prediction(graph, requirements, constraints)
        
        # Create and register the team
        team = await self._create_team(team_composition, task)
        
        # Record collaboration in graph for future learning
        await self.graph_service.record_team_formation(team, task)
        
        return team
        
    @log_execution_time
    async def evaluate_team_performance(self, team: Team, metrics: TeamPerformanceMetrics) -> None:
        """
        Evaluate team performance and update the GNN model weights to improve future predictions.
        This function is critical for the evolutionary aspect of agent collaboration.
        """
        # Record performance metrics in graph
        await self.graph_service.record_performance_metrics(team, metrics)
        
        # Extract features for training
        features = await self._extract_training_features(team, metrics)
        
        # Update model weights
        await self._update_gnn_model(features, metrics.success_score)
        
        # Generate improvement recommendations
        recommendations = self._generate_improvement_recommendations(team, metrics)
        
        # Apply automatic optimizations
        if self.config.enable_auto_optimization:
            await self._apply_optimizations(team, recommendations)
            
    async def _extract_task_requirements(self, task: Task) -> list[Capability]:
        """Extract required capabilities from a task description."""
        ...
        
    def _apply_gnn_prediction(
        self, 
        graph: CapabilityGraph, 
        requirements: list[Capability],
        constraints: TeamConstraints
    ) -> TeamComposition:
        """Apply the GNN model to predict optimal team composition."""
        ...
        
    async def _create_team(self, composition: TeamComposition, task: Task) -> Team:
        """Create and register a new team based on predicted composition."""
        ...
```

### Graph Database Schema

Our Neo4j database schema is optimized for agent capability representation, relationship tracking, and GNN operations:

```
(Agent)-[:HAS_CAPABILITY]->(Capability)
(Agent)-[:COLLABORATED_WITH {success_rate: float}]->(Agent)
(Team)-[:INCLUDES]->(Agent)
(Team)-[:COMPLETED]->(Task)
(Task)-[:REQUIRES]->(Capability)
(Capability)-[:RELATED_TO {strength: float}]->(Capability)
(Agent)-[:EVOLVED_FROM]->(Agent)
(TeamFormation)-[:USED_PATTERN]->(CollaborationPattern)
```

### Interaction Flow: Task to Team Assembly

1. **Task Analysis**: When a new task is registered, capabilities are extracted using NLP
2. **Graph Query**: The system queries the graph for agents with matching capabilities
3. **GNN Prediction**: The GNN predicts the optimal team based on historical performance
4. **Cold Start Integration**: For new agents, the GNN identifies suitable capability inheritance
5. **Team Formation**: The recommended team is assembled with clear communication channels
6. **Execution & Learning**: As the team works, the system records interactions and outcomes
7. **Evolutionary Feedback**: Performance metrics feed back into the GNN for continuous improvement

## Testing & Quality Assurance

Our development approach emphasizes:

1. **Test-Driven Development**
   - 100% test coverage for core components
   - Mocked dependencies for isolated unit testing
   - Integration tests with Neo4j test containers

2. **Property-Based Testing**
   - Graph structure validation
   - Collaboration pattern consistency
   - Agent capability coherence

3. **Performance Benchmarking**
   - GNN inference speed optimization
   - Neo4j query performance tracking
   - End-to-end request timing

## Extensibility & Integration

Clubhouse is designed for extensibility with:

1. **Tool Integration API**
   - OpenAPI specification
   - Automatic capability registration
   - Authentication and authorization controls

2. **Tool Building Process**
   - API schema discovery
   - Automatic wrapper generation
   - Tool registration in capability graph

3. **Model Provider Interface**
   - Standardized protocol for LLM integration
   - Usage tracking and cost accounting
   - Performance benchmarking

## Security & Compliance

Our security model provides:

1. **Capability-Based Control**
   - Fine-grained permission model
   - Least-privilege access to tools and data
   - Capability attestation and verification

2. **Human-in-the-Loop Approval**
   - Critical action approval workflows
   - Transition authorization model
   - Audit logging for all approvals

3. **Privacy-Preserving Design**
   - Data minimization principles
   - Configurable retention policies
   - Compliance with major regulations
