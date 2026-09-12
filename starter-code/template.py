"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Ghi chú: Lab này không cấu hình API key cho LLM thật (xem requirements.txt),
nên "bộ não suy luận" của ReActAgent được mô phỏng bằng luật (rule-based)
để bám sát định dạng Thought / Action / Observation / Final Answer mô tả
trong SYSTEM_PROMPT bên dưới, đồng thời vẫn thể hiện đầy đủ cơ chế ReAct
loop, parse Action JSON, và safeguard max_iterations theo student_guide.md.
"""

import json
import re
import sys
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

# Một số console Windows (cp1252) không in được tiếng Việt có dấu -> ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

AIRPORT_CODES = ["HAN", "SGN", "DAD"]
FAQ_KEYWORDS = ["chính sách", "đổi trả", "hoàn tiền", "hoàn vé"]


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def query(self, user_input: str) -> dict:
        # Không có tool/database nào được gọi ở đây -> chatbot chỉ có thể
        # trả lời bằng kiến thức tổng quát và thừa nhận giới hạn của nó,
        # thay vì bịa ra số hiệu chuyến bay hay số liệu thời tiết cụ thể.
        answer = (
            "Xin lỗi, tôi không có quyền truy cập trực tiếp vào hệ thống đặt vé "
            "hay dữ liệu thời tiết thời gian thực, nên tôi không thể tra cứu "
            f"chính xác cho yêu cầu: \"{user_input}\". "
            "Vui lòng sử dụng ReAct Agent (có tích hợp tool) để có thông tin chính xác."
        )
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],
        }


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    def run(self, user_input: str) -> dict:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        context = {}
        iteration = 0

        # TODO 3 (một phần): "Thought" đầu tiên quyết định kế hoạch gọi tool
        tools_plan = self._plan_tools(user_input)

        # Trường hợp không cần tool nào (FAQ hoặc câu hỏi chung chung)
        if not tools_plan:
            iteration = 1
            thought = "Câu hỏi này không cần tra cứu tool nào, tôi sẽ trả lời trực tiếp."
            answer = self._direct_answer(user_input)
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": None,
                "observation": None,
                "final_answer": answer,
            })
            return {
                "status": "completed",
                "iterations": iteration,
                "trace": self.trace,
                "answer": answer,
            }

        # TODO 2: Vòng lặp while/for iteration < self.max_iterations
        for tool_name in tools_plan:
            iteration += 1

            # TODO 5 (safeguard - Milestone 4 / Trap 3): dừng nếu vượt quá max_iterations
            if iteration > self.max_iterations:
                return {
                    "status": "max_iterations_reached",
                    "answer": "Không thể hoàn thành trong số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration - 1,
                }

            thought = f"Tôi cần dữ liệu thực tế nên sẽ gọi tool '{tool_name}'."
            args = self._build_args(tool_name, user_input)
            action = {"name": tool_name, "args": args}

            # TODO 3: Phân tích Action JSON (Trap 2: json.loads trong try/except)
            action_json_str = json.dumps(action, ensure_ascii=False)
            try:
                parsed_action = json.loads(action_json_str)
            except json.JSONDecodeError:
                observation = {"error": "Invalid JSON format"}
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action_json_str,
                    "observation": observation,
                })
                continue

            # TODO 4: Thực thi Tool trong TOOL_MAP (Trap 1: strip().lower() tên tool)
            clean_name = str(parsed_action.get("name", "")).strip().lower()
            tool_fn = TOOL_MAP.get(clean_name)
            tool_args = parsed_action.get("args", {})
            if tool_fn is None:
                observation = {"error": f"Unknown tool: {clean_name}"}
            else:
                try:
                    observation = tool_fn(**tool_args)
                except Exception as exc:  # noqa: BLE001 - ghi nhận lỗi tool vào Observation
                    observation = {"error": str(exc)}

            context[clean_name] = observation
            # TODO 5: Ghi lại Observation vào self.trace
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": parsed_action,
                "observation": observation,
            })

        # Nếu chỉ cần đúng 1 tool, tổng hợp Final Answer ngay trong bước cuối
        if len(tools_plan) == 1:
            answer = self._synthesize_final_answer(context)
            self.trace[-1]["final_answer"] = answer
            return {
                "status": "completed",
                "iterations": iteration,
                "trace": self.trace,
                "answer": answer,
            }

        # Nhiều tool -> cần thêm 1 bước Thought/Final Answer để tổng hợp
        iteration += 1
        if iteration > self.max_iterations:
            return {
                "status": "max_iterations_reached",
                "answer": "Không thể hoàn thành trong số bước tối đa.",
                "trace": self.trace,
                "iterations": iteration - 1,
            }

        thought = "Tôi đã thu thập đủ dữ liệu từ các tool, giờ tổng hợp câu trả lời cuối cùng."
        answer = self._synthesize_final_answer(context)
        self.trace.append({
            "iteration": iteration,
            "thought": thought,
            "action": None,
            "observation": None,
            "final_answer": answer,
        })
        return {
            "status": "completed",
            "iterations": iteration,
            "trace": self.trace,
            "answer": answer,
        }

    # ------------------------------------------------------------------
    # Helpers: "Thought" (planning) & trích xuất tham số
    # ------------------------------------------------------------------
    def _plan_tools(self, user_input: str) -> list:
        lower = user_input.lower()
        codes_found = re.findall(r"\b(HAN|SGN|DAD)\b", user_input.upper())

        candidates = []

        flight_kw_pos = lower.find("chuyến bay")
        if flight_kw_pos == -1 and "vé" in lower and len(codes_found) >= 1:
            flight_kw_pos = lower.find("vé")
        if flight_kw_pos != -1 and len(codes_found) >= 1:
            candidates.append((flight_kw_pos, "get_flight_info"))

        weather_kw_pos = -1
        for kw in ["thời tiết", "mặc gì", "nhiệt độ"]:
            pos = lower.find(kw)
            if pos != -1 and (weather_kw_pos == -1 or pos < weather_kw_pos):
                weather_kw_pos = pos
        if weather_kw_pos != -1:
            candidates.append((weather_kw_pos, "get_weather_forecast"))

        candidates.sort(key=lambda item: item[0])
        return [name for _, name in candidates]

    def _build_args(self, tool_name: str, user_input: str) -> dict:
        if tool_name == "get_flight_info":
            origin, destination = self._extract_route(user_input)
            max_price = self._extract_price(user_input)
            args = {}
            if origin:
                args["origin"] = origin
            if destination:
                args["destination"] = destination
            if max_price:
                args["max_price"] = max_price
            return args
        if tool_name == "get_weather_forecast":
            city = self._extract_city(user_input)
            return {"city_code": city} if city else {}
        return {}

    def _extract_route(self, user_input: str):
        codes = re.findall(r"\b(HAN|SGN|DAD)\b", user_input.upper())
        if len(codes) >= 2:
            return codes[0], codes[1]
        return None, None

    def _extract_city(self, user_input: str):
        codes = re.findall(r"\b(HAN|SGN|DAD)\b", user_input.upper())
        return codes[-1] if codes else None

    def _extract_price(self, user_input: str):
        lower = user_input.lower()
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*tri[eệ]u", lower)
        if match:
            value = float(match.group(1).replace(",", "."))
            return int(value * 1_000_000)
        match = re.search(r"(\d+)\s*k\b", lower)
        if match:
            return int(match.group(1)) * 1_000
        return None

    # ------------------------------------------------------------------
    # Helpers: tổng hợp câu trả lời cuối
    # ------------------------------------------------------------------
    def _direct_answer(self, user_input: str) -> str:
        lower = user_input.lower()
        if any(kw in lower for kw in FAQ_KEYWORDS):
            return (
                "Theo chính sách đổi trả vé máy bay hiện hành của Vinpearl, quý khách "
                "có thể yêu cầu đổi lịch hoặc hoàn vé tối thiểu 24 giờ trước giờ khởi "
                "hành, có thể phát sinh phí xử lý tùy hạng vé. Vui lòng liên hệ tổng đài "
                "chăm sóc khách hàng Vinpearl để được hỗ trợ chi tiết."
            )
        return (
            "Xin lỗi, tôi chưa có đủ thông tin hoặc tool phù hợp để trả lời chính xác "
            "câu hỏi này. Bạn có thể cung cấp thêm chi tiết được không?"
        )

    def _synthesize_final_answer(self, context: dict) -> str:
        parts = []

        if "get_flight_info" in context:
            flights = context["get_flight_info"]
            if isinstance(flights, list) and len(flights) > 0:
                flight_lines = "; ".join(
                    f"{fl['flight_number']} ({fl['airline']}, khởi hành {fl['departure_time']}, "
                    f"giá {fl['price_vnd']:,} VNĐ)"
                    for fl in flights
                )
                parts.append(f"Các chuyến bay phù hợp: {flight_lines}.")
            else:
                parts.append("Không tìm thấy chuyến bay phù hợp với yêu cầu của bạn.")

        if "get_weather_forecast" in context:
            weather = context["get_weather_forecast"]
            if isinstance(weather, dict) and "error" not in weather:
                parts.append(
                    f"Thời tiết tại {weather['city']}: {weather['temperature_c']}°C, "
                    f"{weather['condition']}. Gợi ý trang phục: {weather['recommendation']}"
                )
            else:
                error_msg = weather.get("error", "unknown error") if isinstance(weather, dict) else "unknown error"
                parts.append(f"Không có dữ liệu thời tiết: {error_msg}")

        if not parts:
            parts.append("Xin lỗi, tôi chưa thể tổng hợp được câu trả lời cho yêu cầu này.")

        return " ".join(parts)


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
