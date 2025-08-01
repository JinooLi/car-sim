# How to use poetry

## poetry init
`poetry init`
-> `poetry.toml` 파일을 대화형으로 생성


## poetry shell
가상환경 생성 & 실행하는 명령어
-> `poetry.toml`에 해당하는 파이썬 가상환경이 생성 및 실행됨
-> 이때, 가상환경이 제대로 실행되지 않는다면,
`. [poetry 가상환경 경로]/bin/activate` 명령어로 가상환경 켜주기

## poetry install
현재 프로젝트의 `poetry.toml` 파일을 읽어서 의존성 패키지를 설치하는 명령어
-> 이때, `poetry.lock`이 없을 경우 해당 파일을 제작하여 활용하고, 이미 존재할 경우 존재하는 `poetry.lock` 파일을 활용
-> 생성되는 `poetry.lock` 파일을 공유할 경우, 협업 시 의존성을 고려한 개발 환경 공유가 가능해짐

## poetry add
필요 패키지를 추가하는 명령어
```bash
    # 기본 활용 방식
    poetry add torch

    # 버전 지정
    poetry add torch@^1.13.1
```
## poetry update
의존성 패키지를 update하고, poetry.lock파일 또한 update하는 명령어

```bash
    # 기본 활용 방식
    poetry update

    # 특정 패키지만 update 하는 경우
    poetry update torch
```
## poetry remove
```bash
    # 특정 패키지를 삭제하는 명령어
    poetry remove torch
```

## poetry show
설치한 패키지를 보여주는 명령어
```bash
    # 기본 활용 방식
    poetry show

    # 업데이트해야하는 패키지를 보여주는 경우
    poetry show --outdate (-o)

    # 의존성 트리를 보여주는 경우
    poetry show --tree
```
## poetry export
poetry.toml 파일을 다른 형식으로 내보내는 명령어
```bash
    # 기본 활용 방식
    poetry export -f requirements.txt --output requirements.txt

    # 해시 정보 없이 export 하기
    poetry export -f requirements.txt --output requirements.txt --without-hashes
```
## poetry env
- 생성한 가상환경 정보 확인하는 명령어  
`poetry env info`

- 가상환경 리스트 확인하는 명령어  
`poetry env list`

- 가상환경 삭제하는 명령어  
`poetry env remove [가상환경 경로]`


## 출처 
[https://velog.io/@eenzeenee/poetry-%EC%82%AC%EC%9A%A9%EB%B2%95-%EC%A0%95%EB%A6%AC](https://velog.io/@eenzeenee/poetry-%EC%82%AC%EC%9A%A9%EB%B2%95-%EC%A0%95%EB%A6%AC)