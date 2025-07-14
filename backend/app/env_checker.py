import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def check_api_keys():
    """
    필수 API 키들이 환경변수에 설정되어 있는지 체크하는 함수
    """
    print("\n>> API 키 환경변수 체크 시작")
    print("=" * 50)

    # 체크할 API 키 목록
    required_keys = ["OPENAI_API_KEY", "SERPER_API_KEY", "TAVILY_API_KEY"]

    missing_keys = []

    for key in required_keys:
        if not os.environ.get(key):
            print(f"<< {key}: 설정되지 않음")
            missing_keys.append(key)
        else:
            # 키의 일부만 표시 (보안을 위해)
            masked_key = os.environ.get(key)[:8] + "..." + os.environ.get(key)[-4:]
            print(f">> {key}: 설정 완료 ({masked_key})")

    print("=" * 50)

    if missing_keys:
        print(f">> 누락된 환경변수: {', '.join(missing_keys)}")
        print(">> .env 파일에 다음과 같이 추가해주세요:")
        for key in missing_keys:
            print(f"   {key}=your_api_key_here")
        return False
    else:
        print(">> 모든 필수 API 키가 설정되었습니다!")
        return True


def set_environment_variables():
    """
    환경변수들을 설정하는 함수
    """
    # 각 API 키를 환경변수에서 가져와서 설정
    openai_key = os.environ.get("OPENAI_API_KEY")
    serper_key = os.environ.get("SERPER_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")

    # 전역 환경변수로 설정 (필요한 경우)
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if serper_key:
        os.environ["SERPER_API_KEY"] = serper_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key


if __name__ == "__main__":
    # API 키 체크 실행
    keys_valid = check_api_keys()

    if keys_valid:
        print("\n>> 환경변수 설정 완료 - 프로젝트를 시작할 수 있습니다!")
        set_environment_variables()
    else:
        print("\n>> 환경변수 설정을 완료한 후 다시 실행해주세요.")
