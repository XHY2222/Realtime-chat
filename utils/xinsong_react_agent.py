import asyncio
import json
from typing import Dict, List, Optional


class XinsongReactAgent:
    def __init__(
        self,
        model_name: str = "qwen-plus",
        api_key: str = "",
        api_base: str = "",
        use_agentscope_react: bool = True,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.use_agentscope_react = use_agentscope_react

    def _build_prompt(self, query: str, retrievals: List[Dict]) -> str:
        evidence_blocks = []
        for i, item in enumerate(retrievals, start=1):
            evidence_blocks.append(
                f"[{i}] 来源: {item.get('source', '未知')}\n内容: {item.get('text', '')}"
            )
        evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "无"

        return (
            "你是新松机器人领域问答助手。\n"
            "请采用 ReAct 思路进行内部推理（不要输出思考过程），最终只输出 JSON。\n"
            "任务：\n"
            "1) 识别用户意图(intent)\n"
            "2) 仅基于给定证据形成回答，不得编造\n"
            "3) 输出结构: {\"intent\":\"...\",\"evidence\":[\"...\"],\"final_answer\":\"...\"}\n\n"
            f"用户问题:\n{query}\n\n"
            f"检索证据:\n{evidence_text}\n"
        )

    async def _call_openai_compatible(self, prompt: str) -> Optional[str]:
        try:
            import openai
            from packaging import version

            if version.parse(openai.__version__) < version.parse("1.0.0"):
                openai.api_base = self.api_base
                openai.api_key = self.api_key
                response = await asyncio.to_thread(
                    openai.ChatCompletion.create,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是严谨的企业知识问答助手"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    stream=False,
                )
                return response["choices"][0]["message"]["content"]

            client = openai.OpenAI(base_url=self.api_base, api_key=self.api_key)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是严谨的企业知识问答助手"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                stream=False,
            )
            if response.choices:
                return response.choices[0].message.content
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_text_from_agentscope_result(result) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("content", "text", "output", "response"):
                if key in result and result[key]:
                    return str(result[key])
            return ""
        getter = getattr(result, "get_text_content", None)
        if callable(getter):
            try:
                return getter() or ""
            except Exception:
                return ""
        content = getattr(result, "content", None)
        if content:
            return str(content)
        return str(result)

    async def _run_with_agentscope(self, query: str, retrievals: List[Dict]) -> Optional[str]:
        if not self.use_agentscope_react:
            return None

        try:
            from agentscope.agent import ReActAgent
            from agentscope.formatter import DashScopeChatFormatter
            from agentscope.memory import InMemoryMemory
            from agentscope.model import DashScopeChatModel
            from agentscope.tool import Toolkit
        except Exception:
            return None

        evidence_blocks = []
        for i, item in enumerate(retrievals, start=1):
            evidence_blocks.append(
                f"[{i}] 来源: {item.get('source', '未知')}\n内容: {item.get('text', '')}"
            )
        evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "无"

        toolkit = Toolkit()

        def search_xinsong_kb(_: str = "") -> str:
            return evidence_text

        toolkit.register_tool_function(search_xinsong_kb)

        agent = ReActAgent(
            name="XinsongReAct",
            sys_prompt=(
                "你是新松机器人问答助手。"
                "请使用 ReAct 方式完成内部推理并调用工具，但不要输出思考过程。"
                "最终仅输出 JSON: {\"intent\":\"...\",\"evidence\":[\"...\"],\"final_answer\":\"...\"}"
            ),
            model=DashScopeChatModel(
                api_key=self.api_key,
                model_name=self.model_name,
                enable_thinking=False,
                stream=False,
            ),
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )

        prompt = (
            f"用户问题：{query}\n"
            "请先调用 search_xinsong_kb 获取证据，再给出最终 JSON 输出。"
        )
        result = await agent(prompt)
        return self._extract_text_from_agentscope_result(result)

    def _fallback_result(self, query: str, retrievals: List[Dict]) -> Dict:
        intent = "公司概况"
        q = query or ""
        if "产品" in q or "机器人" in q:
            intent = "产品体系"
        elif "历程" in q or "发展" in q or "哪年" in q:
            intent = "发展历程"
        elif "技术" in q or "优势" in q:
            intent = "技术优势"
        elif "最新" in q or "动态" in q:
            intent = "最新动态"

        evidence = []
        for item in retrievals[:3]:
            text = (item.get("text") or "").strip().replace("\n", " ")
            if len(text) > 110:
                text = text[:110] + "..."
            evidence.append(f"{item.get('source', '未知')}: {text}")

        if evidence:
            final_answer = "根据新松知识库，" + "；".join(evidence)
        else:
            final_answer = "我在当前新松知识库中没有检索到足够信息，请你换个问法再试试。"

        return {
            "intent": intent,
            "evidence": evidence,
            "final_answer": final_answer,
        }

    async def run(self, query: str, retrievals: List[Dict]) -> Dict:
        content = await self._run_with_agentscope(query, retrievals)
        if not content:
            prompt = self._build_prompt(query, retrievals)
            content = await self._call_openai_compatible(prompt)
        if not content:
            return self._fallback_result(query, retrievals)

        try:
            if "```" in content:
                content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            if not isinstance(data, dict):
                return self._fallback_result(query, retrievals)

            intent = data.get("intent") or "新松问答"
            evidence = data.get("evidence") or []
            final_answer = data.get("final_answer") or ""
            if not final_answer:
                return self._fallback_result(query, retrievals)

            return {
                "intent": str(intent),
                "evidence": evidence if isinstance(evidence, list) else [str(evidence)],
                "final_answer": str(final_answer),
            }
        except Exception:
            return self._fallback_result(query, retrievals)
