from ..base import MemoryProviderBase, logger
from mem0 import MemoryClient

TAG = __name__

class MemoryProvider(MemoryProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.api_version = config.get("api_version", "v1.1")
        if len(self.api_key) == 0 or "你" in self.api_key:
            logger.bind(tag=TAG).error("你还没配置Mem0ai的密钥，请在配置文件中配置密钥，否则无法提供记忆服务")
            self.use_mem0 = False
            return
        else:
            self.use_mem0 = True
        self.client = MemoryClient(api_key=self.api_key)

    async def save_memory(self, msgs):
        if not self.use_mem0:
            return None
        if len(msgs) < 2:
            return None
        
        try:
            # Format the content as a message list for mem0
            messages = [
                {"role": message.role, "content": message.content}
                for message in msgs if message.role != "system"
            ]
            result = self.client.add(messages, user_id=self.role_id, output_format=self.api_version)
            logger.bind(tag=TAG).debug(f"Save memory result: {result}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存记忆失败: {str(e)}")
            return None

    async def query_memory(self, query: str)-> str:
        if not self.use_mem0:
            return ""
        try:
            results = self.client.search(
                query,
                user_id=self.role_id,
                output_format=self.api_version
            )
            if not results or 'results' not in results:
                return ""
                
            # Format each memory entry with its update time up to minutes
            memories = []
            for entry in results['results']:
                # Split timestamp and get date + time up to minutes
                timestamp = entry.get('updated_at', '').split('.')[0]  # Remove milliseconds
                if timestamp:
                    try:
                        # Parse and reformat the timestamp
                        dt = timestamp.replace('T', ' ').split(':')  # Split time components
                        formatted_time = f"{dt[0]}:{dt[1]}:{dt[2]}"  # Keep only HH:MM:SS
                    except:
                        formatted_time = timestamp
                memory = entry.get('memory', '')
                if timestamp and memory:
                    memories.append(f"[{formatted_time}] {memory}")
                    
            memories_str = "\n".join(f"- {memory}" for memory in memories)
            logger.bind(tag=TAG).debug(f"Query results: {memories_str}")
            return memories_str
        except Exception as e:
            logger.bind(tag=TAG).error(f"查询记忆失败: {str(e)}")
            return ""