import os
import pathlib
import time
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

ROOT = pathlib.Path(__file__).parent.parent

# =========================================================================
# 1. 가상 데이터베이스 및 로그 유틸리티 정의 (현장 환경 시뮬레이션용)
# =========================================================================

def _get_mock_flight_schedule(unit_id: int) -> int:
    """
    [시뮬레이션 모사] 대상 엔진의 다음 비행 일정(이륙 예정 시간)까지 남은 사이클 수를 반환합니다.
    실제 현장에서는 항공 스케줄 DB를 조회하는 API 호출과 매핑됩니다.
    """
    return (unit_id * 7 + 13) % 25 + 2

def _get_mock_contact(unit_id: int) -> dict:
    """
    [시뮬레이션 모사] 대상 엔진이 배치된 정비구역(Zone)과 담당 정비 책임자의 인적사항을 반환합니다.
    에이전트가 어떤 정비사에게 연락할지 대상자(이름, 전화번호, 슬랙채널)를 식별하기 위해 사용됩니다.
    """
    zones = ["A", "B", "C"]
    zone = zones[unit_id % 3]
    managers = ["김정비 과장", "이대리 대리", "박팀장 팀장"]
    manager = managers[unit_id % 3]
    phone = f"010-9999-{1000 + unit_id:04d}"
    slack_channel = f"#mro-zone-{zone.lower()}"
    return {
        "manager": manager,
        "phone": phone,
        "slack_channel": slack_channel,
        "zone": zone
    }

def _log_alert(message: str):
    """
    [로그 보존] 외부로 발신된 알람 이력을 'reports/alerts.log' 파일에 영구 기록합니다.
    """
    log_dir = ROOT / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "alerts.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


# =========================================================================
# 2. LangChain 에이전트 전용 Tools 정의 (자율적 수단 판단용)
# =========================================================================

@tool
def check_current_time_context(tick: int) -> str:
    """현재 시뮬레이션 틱(tick)을 기준으로 야간 비근무 시간대(22:00 ~ 08:00)인지 주간 교대 시간대인지 판별합니다."""
    minutes_in_day = tick % 1440
    hours = minutes_in_day // 60
    mins = minutes_in_day % 60
    time_str = f"{hours:02d}:{mins:02d}"
    
    # 22:00 (1320분) 이후 혹은 08:00 (480분) 이전은 야간 비출근 시간대로 규정
    if minutes_in_day >= 1320 or minutes_in_day < 480:
        return f"현재 시각 {time_str} [NIGHT_OFF_HOURS] (야간 비출근 시간대 - 현장 정비사 수면 중. 절대 긴급한 경우에만 수신 거부 해제용 전화 발신 권장)"
    else:
        return f"현재 시각 {time_str} [DAY_SHIFT] (주간 교대 시간대 - 전체 근무 중)"

@tool
def get_next_flight_schedule(unit_id: int) -> str:
    """대상 엔진 유닛의 다음 이륙(비행) 스케줄까지 남은 예상 시간(사이클 수)을 가상 스케줄러 DB에서 조회합니다."""
    remaining_cycles = _get_mock_flight_schedule(unit_id)
    return f"엔진 #{unit_id} 다음 비행 예정: {remaining_cycles} 사이클 남음."

@tool
def get_fleet_contacts(unit_id: int) -> str:
    """대상 엔진이 속한 정비구역(Zone)과 담당 정비 책임자의 성함, 휴대폰 번호 및 슬랙 채널을 조회합니다."""
    contact = _get_mock_contact(unit_id)
    return str(contact)

@tool
def trigger_emergency_slack(unit_id: int, slack_channel: str, message: str) -> str:
    """근무 중인 정비조 전체가 볼 수 있도록 공식 메신저 슬랙(Slack) 채널로 진단 에러 상황 및 작업지시 대기를 브로드캐스팅합니다."""
    alert_msg = f"[🚨 EMERGENCY SLACK ALERT] [Channel: {slack_channel}] 엔진 #{unit_id} 경보: {message}"
    print(alert_msg)
    _log_alert(alert_msg)
    return "슬랙 비상 알림 발신 성공"

@tool
def trigger_tts_voice_call(unit_id: int, phone_number: str, manager_name: str, message: str) -> str:
    """야간 숙면 중인 담당자 휴대폰 번호로 비상 자동 전화(TTS Voice Call)를 강제 발신하여 정비사를 즉각 깨고 음성 지침을 안내합니다."""
    alert_msg = f"[📞 TTS VOICE CALL INITIATED] [Recipient: {manager_name} ({phone_number})] 지침 전달: {message}"
    print(alert_msg)
    _log_alert(alert_msg)
    return "정비사 비상 유선 전화 자동 연결 및 TTS 지침 안내 완료"


# =========================================================================
# 3. SmartAlertAgent 클래스 구현 (알람 피로도 관리 적용)
# =========================================================================

class SmartAlertAgent:
    """
    [지능형 가드 에이전트]
    센서 전이 및 RUL 급락 발생 시 알람 피로도(Alert Fatigue)를 원천 차단하기 위해
    1. 'inspect(점검요망)'와 같은 경고 수준은 오직 대시보드 로컬 로그에만 출력하도록 통제하며,
    2. 오직 'danger(위험)' 수준의 치명적 경보일 때만 시각(낮/밤) 및 다음 비행의 긴급성을 종합 평가해 
       슬랙 전송 또는 자동 TTS 유선 전화를 차등 발신합니다.
    """
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.tools = [
            check_current_time_context,
            get_next_flight_schedule,
            get_fleet_contacts,
            trigger_emergency_slack,
            trigger_tts_voice_call
        ]
        
        # 알람 피로 방지 규칙이 반영된 시스템 프로토콜 정의
        self.system_prompt = """당신은 항공 제트엔진 MRO 관제소의 지능형 알람 필터링 에이전트(SmartAlertAgent)입니다.
당신의 임무는 엔진 상태 이탈 정보(status, RUL)를 바탕으로, 현장 작업자의 피로도를 최소화하며 긴급성을 제어하는 것입니다.

[필수 의사결정 프로토콜]
1. 먼저 들어온 이상 상태(status)가 'inspect'(경고/점검요망)인지 'danger'(위험/고장임박)인지 확인하십시오.
   - [피로 방지 규칙 - inspect 상태인 경우]:
     정비사를 방해해서는 안 됩니다. 슬랙이나 유선전화 도구를 절대 호출하지 마십시오.
     단순히 대시보드 노출을 위해 "[대시보드 로그 전용] 엔진 #X 점검요망 상태 감지 - 외부 발신 생략." 포맷의 최종 답변만 리턴하고 강제 종료하십시오.
   
2. 상태가 'danger'인 경우 아래 도구들을 사용해 수위를 판단하십시오:
   - `check_current_time_context`로 야간(NIGHT_OFF_HOURS)인지 주간(DAY_SHIFT)인지 판별하십시오.
   - `get_next_flight_schedule`로 정비 시점까지 남은 비행 스케줄(사이클) 여유를 확인하십시오.
   - `get_fleet_contacts`로 수신할 담당 정비사의 연락처를 확인하십시오.
   
3. 'danger' 경보의 전송 수단 분기법:
   - [주간(DAY_SHIFT)인 경우]: 
     근무 중이므로 슬랙 공식 알람 도구(`trigger_emergency_slack`)만 발신하십시오.
   - [야간(NIGHT_OFF_HOURS) & 다음 비행까지 여유(5 사이클 초과)가 있는 경우]: 
     자고 있는 정비사를 깨우지 않고, 출근 후 볼 수 있도록 슬랙 공식 알람 도구(`trigger_emergency_slack`)만 발신하십시오.
   - [야간(NIGHT_OFF_HOURS) & 다음 비행이 임박(5 사이클 이하)한 긴급 상황인 경우]: 
     매우 치명적입니다. 잠든 정비사를 즉각 깨워야 하므로 자동 TTS 비상 전화 도구(`trigger_tts_voice_call`)를 호출하십시오.

도구 실행 결과를 얻은 후, 거쳐간 생각(Thought)과 최종 조치 결과를 한글 전문 답변으로 정중히 요약하십시오."""

        self.model_with_tools = None
        if self.api_key:
            try:
                # 최신 langchain-openai의 bind_tools를 사용하는 자율 에이전트 루프 초기화
                chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=self.api_key)
                self.model_with_tools = chat.bind_tools(self.tools)
            except Exception as e:
                print(f"Warning: Failed to initialize LangChain ChatOpenAI: {e}. Running in Static Fallback Mode.")

    def run_alert_logic(self, unit_id: int, tick: int, status: str, rul: float) -> str:
        """
        [주요 진입점]
        이상 감지된 엔진의 정보를 종합해 경보 여부 및 발신 채널을 판단합니다.
        API Key가 활성화되어 있지 않으면 정적 규칙 엔진(Static Fallback)이 작동합니다.
        """
        if not self.model_with_tools:
            return self._run_static_fallback(unit_id, tick, status, rul)
            
        try:
            user_input = f"엔진 #{unit_id}에서 이상 상태 '{status}' 감지 (예측 RUL: {rul:.1f}). 시뮬레이션 틱: {tick}. 최적의 비상 전송 수단을 선별하여 동작시켜 줘."
            
            # ReAct (Reasoning and Action) 루프 수동 전개
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input)
            ]
            
            # 최대 5회 툴 호출 의사결정 루프 전개
            for i in range(5):
                print(f"[SmartAlertAgent] Thinking Step {i+1}...")
                response = self.model_with_tools.invoke(messages)
                messages.append(response)
                
                # AI가 도구 호출 없이 최종 텍스트 답변을 지목했거나 루프 종료 선언 시 리턴
                if not response.tool_calls:
                    return response.content
                    
                # 호출이 지정된 각 도구 실행
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # 매칭되는 툴 탐색 후 자율 실행
                    matching_tool = next((t for t in self.tools if t.name == tool_name), None)
                    if matching_tool:
                        print(f"[SmartAlertAgent] Executing Tool: {tool_name} with args: {tool_args}")
                        tool_output = matching_tool.invoke(tool_args)
                        messages.append(ToolMessage(
                            content=str(tool_output),
                            tool_call_id=tool_call["id"]
                        ))
                    else:
                        messages.append(ToolMessage(
                            content=f"Error: Tool '{tool_name}' not found.",
                            tool_call_id=tool_call["id"]
                        ))
            return "Error: Maximum agent loop iterations exceeded."
            
        except Exception as exc:
            print(f"Warning: LangChain SmartAlertAgent run failed: {exc}. Falling back to static logic.")
            return self._run_static_fallback(unit_id, tick, status, rul)

    def _run_static_fallback(self, unit_id: int, tick: int, status: str, rul: float) -> str:
        """
        [정적 Fallback 규칙 엔진]
        OpenAI API 연결 실패 혹은 미연동 시 로컬 오프라인에서 안전하게 경보 필터를 작동시킵니다.
        """
        # 피로 방지 규칙 1: 단순 inspect 상태는 외부로 전송하지 않고 로컬 대시보드 로깅만 수행
        if status == "inspect":
            return f"[대시보드 로그 전용] 엔진 #{unit_id} 점검요망 상태 감지 - 외부 발신 생략."
            
        # status == "danger" 인 핵심 긴급 상황
        minutes_in_day = tick % 1440
        is_night = minutes_in_day >= 1320 or minutes_in_day < 480
        remaining_cycles = _get_mock_flight_schedule(unit_id)
        contact = _get_mock_contact(unit_id)
        
        message = f"유닛 #{unit_id} 위험 상태 '{status}' 이탈 (RUL: {rul:.1f})"
        
        if is_night and remaining_cycles <= 5:
            # 야간 초긴급 상황 -> 유선전화 TTS 호출
            trigger_tts_voice_call.invoke({
                "unit_id": unit_id,
                "phone_number": contact["phone"],
                "manager_name": contact["manager"],
                "message": message
            })
            return f"[Fallback] 야간 초긴급 전화 호출 완료 (대상: {contact['manager']})"
        else:
            # 주간 혹은 야간이어도 다음 비행까지 여유가 있는 경우 -> 슬랙 알림만 브로드캐스트
            trigger_emergency_slack.invoke({
                "unit_id": unit_id,
                "slack_channel": contact["slack_channel"],
                "message": message
            })
            return f"[Fallback] 업무 공식 슬랙 전송 완료 (채널: {contact['slack_channel']})"
