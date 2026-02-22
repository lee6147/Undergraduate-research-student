# Obsidian에서 HTML 파일 열어보기

## 이게 뭔가요?

Obsidian은 기본적으로 `.html` 파일을 열 수 없습니다.
이 레포지터리를 포크하면 **별도 설정 없이** Obsidian에서 HTML 가이드 파일을 바로 열어볼 수 있습니다.

---

## 어떻게 가능한가요?

`.obsidian/plugins/` 폴더에 아래 플러그인이 미리 포함되어 있기 때문입니다.

| 플러그인 | 역할 |
|---------|------|
| **obsidian-html-plugin** | `.html` 파일을 Obsidian 파일 탐색기에서 인식 → 클릭으로 바로 열기 |
| **obsidian-custom-frames** | HTML 파일을 좌측 리본 아이콘 바로가기로 등록 (선택) |

> 핵심은 `obsidian-html-plugin`입니다. 이 플러그인이 없으면 Obsidian은 HTML 파일을 무시합니다.

---

## 포크 후 설정 방법

### 1단계: 레포 클론

```bash
git clone https://github.com/lee6147/Undergraduate-research-student.git
```

### 2단계: Obsidian에서 볼트 열기

1. Obsidian 실행
2. **다른 보관함 열기** → **폴더를 보관함으로 열기**
3. 클론한 폴더 내 `Claude/Obsidian` 경로를 선택

### 3단계: 커뮤니티 플러그인 활성화

처음 열면 커뮤니티 플러그인이 비활성화 상태일 수 있습니다.

1. **설정**(좌측 하단 톱니바퀴) → **커뮤니티 플러그인**
2. **제한 모드 해제** 클릭
3. 이미 설치된 플러그인 목록에서 **HTML Reader** 활성화 확인

### 4단계: HTML 파일 열기

파일 탐색기에서 `Guide_HTML/` 폴더 안의 `.html` 파일을 클릭하면 바로 열립니다.

---

## HTML 파일 추가하기

`Guide_HTML/` 폴더에 `.html` 파일을 넣기만 하면 됩니다.
경로 등록이나 설정 변경 없이 파일 탐색기에 자동으로 나타납니다.

---

## HTML 가이드는 어떻게 만드나요?

이 레포의 [`obsidian-html-react-Prompt.md`](./obsidian-html-react-Prompt.md)를 참고하세요.
AI에게 해당 프롬프트를 붙여넣고 JSX/TSX 파일을 주면 Obsidian에서 실행 가능한 단일 HTML 파일로 변환해줍니다.

---

## 참고 자료

- [obsidian-html-plugin (GitHub)](https://github.com/nuthrash/obsidian-html-plugin) — HTML Reader 플러그인 원본
- [obsidian-vault 레포](https://github.com/lee6147/obsidian-vault) — 초기 Obsidian 세팅 참고용 (플러그인 포함)
