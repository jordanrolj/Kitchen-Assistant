import os
from langchain.prompts import PromptTemplate  # Correct import for PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.chains.llm_math.base import LLMMathChain
from dotenv import load_dotenv
from langchain.tools import tool
from nutrition_api import get_nutrition_info  # Import nutrition function from separate file
from rag import RagEngine
from langchain import hub


load_dotenv()


class ConversationalAgent:
    def __init__(self):
        # Set up the OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize memory for chat history
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize LLM (e.g., GPT-4)
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0,
            model_name="gpt-4",
        )
        
        # Define the system message
        self.system_message = "You are a helpful assistant that suggests recipes, provides nutrition information, and responds to chat history."
        
        # Initialize the RAG engine once during agent initialization
        self.rag_engine = RagEngine()

        # Define the LLMMathChain tool
        llm_math = LLMMathChain(llm=self.llm)
        self.math_tool = Tool(
            name='Calculator',
            func=llm_math.run,
            description='Useful for performing math calculations.'
        )
        
        # Initialize the agent
        self.agent_executor = self.create_agent_executor()

    def create_agent_executor(self):
        # Pull a React prompt template from hub
        prompt = hub.pull("hwchase17/react")

        # Define tools (QA, RAG, Nutrition, Math), using lambda to allow access to instance methods
        tools = [
            Tool(
                name="QA Chat",
                func=lambda message: self.qa_chat_tool(message),
                description="Handles general Q&A, including chat history"
            ),
            Tool(
                name="RAG Chat",
                func=lambda message: self.rag_chat_tool(message),
                description="Finds recipes using internal documents"
            ),
            Tool(
                name="Nutrition Info",
                func=lambda food_item: self.nutrition_tool(food_item),
                description="Fetches nutritional information for a given food item"
            ),
            self.math_tool  # Math tool is already correctly defined
        ]

        # Create the react agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )

        # Create an AgentExecutor to run the agent
        return AgentExecutor.from_agent_and_tools(agent, tools=tools, verbose=True)

    def qa_chat_tool(self, message: str) -> str:
        """Tool for handling general question-answer interactions, including chat history references."""
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        relevant_context = "\n".join([msg['content'] for msg in chat_history[-10:]]) if chat_history else ""
        
        template = """Use the following context to respond to the user's message if relevant:
            context: {relevant_context}
            message: {message}
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke(message=message, relevant_context=relevant_context)
        
        return response

    def rag_chat_tool(self, input_message: str) -> str:
        """Tool for finding recipes using internal documents."""
        return self.rag_engine.return_answer(input_message)

    def nutrition_tool(self, food_item: str) -> str:
        """Tool for fetching nutritional information via an external API."""
        return get_nutrition_info(food_item)

    def handle_input(self, message: str) -> str:
        """Main method to handle user input and invoke the agent."""
        # Invoke the agent executor with the user input and return the output
        return self.agent_executor.invoke({"input": message})
