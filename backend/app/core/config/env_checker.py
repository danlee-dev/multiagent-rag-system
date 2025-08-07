import os
from dotenv import load_dotenv




def check_api_keys():
    """
    필수 API 키들이 환경변수에 설정되어 있는지 체크하는 함수
    """
    print("\n>> API 키 환경변수 체크 시작")
    print("=" * 50)
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

    required_keys = ["OPENAI_API_KEY", "SERPER_API_KEY", "LANGSMITH_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY"]

    missing_keys = []

    for key in required_keys:
        if not os.environ.get(key):
            print(f"<< {key}: 설정되지 않음")
            missing_keys.append(key)
        else:
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

    openai_key = os.environ.get("OPENAI_API_KEY")
    serper_key = os.environ.get("SERPER_API_KEY")
    langsmith_key = os.environ.get("LANGSMITH_API_KEY")


    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if serper_key:
        os.environ["SERPER_API_KEY"] = serper_key
    if tavily_key:
        os.environ["LANGSMITH_API_KEY"] = langsmith_key


if __name__ == "__main__":
    keys_valid = check_api_keys()

    if keys_valid:
        print("\n>> 환경변수 설정 완료 - 프로젝트를 시작할 수 있습니다!")
        set_environment_variables()
    else:
        print("\n>> 환경변수 설정을 완료한 후 다시 실행해주세요.")
