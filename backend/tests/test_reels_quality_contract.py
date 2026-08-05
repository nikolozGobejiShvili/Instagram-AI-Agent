import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.api.routes import agent as agent_route  # noqa: E402
from app.api.routes import internal_generation_debug as internal_generation_debug_route  # noqa: E402
from app.api.routes import knowledge_packs as knowledge_pack_route  # noqa: E402
from app.services.agent_response_formatter_service import AgentResponseFormatterService  # noqa: E402
from app.services.generation_history_service import GenerationHistoryService  # noqa: E402
from app.services.knowledge_pack_service import KnowledgePackService  # noqa: E402
from app.services.langflow_service import LangflowService  # noqa: E402


INTERNAL_HEADERS = {"X-Internal-Admin-Key": "test-internal-key"}


def _build_reel_idea_reply() -> str:
    structured_output = {
        "ideas": [
            {
                "title": "CTA Audit Reel",
                "hook": "If people watch but do not DM, the CTA is usually the problem.",
                "format_type": "Talking-head diagnosis with proof screenshots",
                "main_idea": "Break down one CTA mistake that kills qualified DMs and show the exact rewrite in one sentence.",
                "shot_list": [
                    "Direct-to-camera hook",
                    "Screenshot of weak CTA",
                    "Screenshot of stronger CTA",
                    "Final direct CTA frame",
                ],
                "why_it_can_work": "It creates tension fast, shows proof early, and turns a hidden conversion mistake into a saveable lesson.",
                "cta": "DM me 'CTA' and I will rewrite one of yours.",
            },
            {
                "title": "Trend Adaptation Reel",
                "hook": "A trend only works when it lands on the right pain point for your niche.",
                "format_type": "Trend audio plus myth-versus-reality text",
                "main_idea": "Take a familiar trend pattern and rewrite the text so it exposes one repeated sales-blocking mistake in the niche.",
                "shot_list": [
                    "Trend opening beat",
                    "Myth text overlay",
                    "Reality text overlay",
                    "Fast proof example",
                    "Direct CTA close",
                ],
                "why_it_can_work": "It keeps the trend recognizable while making the message feel native to the account's audience and goal.",
                "cta": "DM me 'TREND' and I will adapt one for your niche.",
            },
            {
                "title": "Comment-to-Reel Conversion",
                "hook": "If this question appears in your comments often, your next Reel is already waiting there.",
                "format_type": "Comment-reply Reel with screen record and voiceover",
                "main_idea": "Turn one repeated objection into a quick Reel that answers the question, shows proof, and leads into a DM CTA.",
                "shot_list": [
                    "Show the original comment",
                    "Voiceover answer opening",
                    "Proof or demo moment",
                    "Single-step takeaway",
                    "DM CTA close",
                ],
                "why_it_can_work": "It starts from a real audience signal, improves relevance, and makes the CTA feel like a natural next step.",
                "cta": "DM me your most common question and I will turn it into a Reel.",
            },
        ]
    }
    return "\n".join([
        "Here are 3 strong Reel ideas for the current Instagram context and lead-generation goal.",
        "",
        "### Reel 1",
        "Title:",
        "CTA Audit Reel",
        "1-second hook:",
        "If people watch but do not DM, the CTA is usually the problem.",
        "Format type:",
        "Talking-head diagnosis with proof screenshots",
        "Main idea:",
        "Break down one CTA mistake that kills qualified DMs and show the exact rewrite in one sentence.",
        "Shot list:",
        "1. Direct-to-camera hook",
        "2. Screenshot of weak CTA",
        "3. Screenshot of stronger CTA",
        "4. Final direct CTA frame",
        "Why it can work:",
        "It creates tension fast, shows proof early, and turns a hidden conversion mistake into a saveable lesson.",
        "CTA:",
        "DM me 'CTA' and I will rewrite one of yours.",
        "",
        "### Reel 2",
        "Title:",
        "Trend Adaptation Reel",
        "1-second hook:",
        "A trend only works when it lands on the right pain point for your niche.",
        "Format type:",
        "Trend audio plus myth-versus-reality text",
        "Main idea:",
        "Take a familiar trend pattern and rewrite the text so it exposes one repeated sales-blocking mistake in the niche.",
        "Shot list:",
        "1. Trend opening beat",
        "2. Myth text overlay",
        "3. Reality text overlay",
        "4. Fast proof example",
        "5. Direct CTA close",
        "Why it can work:",
        "It keeps the trend recognizable while making the message feel native to the account's audience and goal.",
        "CTA:",
        "DM me 'TREND' and I will adapt one for your niche.",
        "",
        "### Reel 3",
        "Title:",
        "Comment-to-Reel Conversion",
        "1-second hook:",
        "If this question appears in your comments often, your next Reel is already waiting there.",
        "Format type:",
        "Comment-reply Reel with screen record and voiceover",
        "Main idea:",
        "Turn one repeated objection into a quick Reel that answers the question, shows proof, and leads into a DM CTA.",
        "Shot list:",
        "1. Show the original comment",
        "2. Voiceover answer opening",
        "3. Proof or demo moment",
        "4. Single-step takeaway",
        "5. DM CTA close",
        "Why it can work:",
        "It starts from a real audience signal, improves relevance, and makes the CTA feel like a natural next step.",
        "CTA:",
        "DM me your most common question and I will turn it into a Reel.",
        "",
        f"STRUCTURED_OUTPUT_JSON: {json.dumps(structured_output, ensure_ascii=False)}",
    ])


def _upload_internal_reels_pack(client: TestClient) -> dict:
    response = client.post(
        "/api/v1/internal/knowledge-packs/upload",
        headers=INTERNAL_HEADERS,
        data={
            "title": "Mariami Reels System Playbook v1",
            "description": "Internal reels methodology for ideas, trend adaptation, hooks, formats, and reel feedback.",
            "domain": "reels",
            "supported_task_types": "reel_idea,reel_script,reel_feedback",
            "scope": "system",
            "visibility": "internal",
            "status": "active",
        },
        files=[
            (
                "files",
                (
                    "mariami-reels.md",
                    b"# KNOWLEDGE MODULE 1 - VIRAL IDEA MECHANICS\nUse tension and simplicity.\n\n# KNOWLEDGE MODULE 2 - TREND STRUCTURE\nAdapt the pattern to the niche and keep the first second sharp.",
                    "text/markdown",
                ),
            ),
        ],
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_reel_idea_formatter_heuristic_fallback_keeps_fields_clean():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "Here are 2 ideas.",
        "",
        "### Reel 1",
        "Title:",
        "CTA Audit Reel",
        "1-second hook:",
        "If people watch but do not DM, the CTA is weak.",
        "Format type:",
        "Talking-head diagnosis",
        "Main idea:",
        "Break down one CTA mistake.",
        "Shot list:",
        "1. Hook",
        "2. Proof",
        "3. Fix",
        "Why it can work:",
        "Fast tension plus proof.",
        "CTA:",
        "DM me 'CTA'.",
        "",
        "### Reel 2",
        "Title:",
        "Comment-to-Reel Conversion",
        "1-second hook:",
        "A repeated audience question is already a Reel idea.",
        "Format type:",
        "Comment reply Reel",
        "Main idea:",
        "Answer one objection fast.",
        "Shot list:",
        "1. Comment",
        "2. Answer",
        "3. CTA",
        "Why it can work:",
        "It feels relevant.",
        "CTA:",
        "DM me your question.",
    ])

    normalized = formatter.normalize_reply("reel_idea", reply)
    assert normalized["parse_status"] == "parsed"
    assert len(normalized["structured_output"]["ideas"]) == 2
    first_idea = normalized["structured_output"]["ideas"][0]
    second_idea = normalized["structured_output"]["ideas"][1]
    assert first_idea["title"] == "CTA Audit Reel"
    assert first_idea["cta"] == "DM me 'CTA'."
    assert "###" not in first_idea["cta"]
    assert "Reel 2" not in first_idea["cta"]
    assert second_idea["title"] == "Comment-to-Reel Conversion"


def test_reel_idea_formatter_parses_idea_blocks_with_caption_ideas():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "IDEA 1",
        "Title:",
        "მონეტიზაცია 100 გამომწერით",
        "Hook:",
        "გგონია შემოსავალი მხოლოდ 10 000 გამომწერიდან იწყება?",
        "Format type:",
        "Talking-head proof Reel",
        "Main idea:",
        "აჩვენე როგორ შეიძლება მცირე აუდიტორიით პირველი შეთავაზების დატესტვა.",
        "Shot list:",
        "1. კამერასთან სწრაფი ჰუკი",
        "2. შედეგის მოკლე proof",
        "3. ერთი actionable ნაბიჯი",
        "Why it can work:",
        "ამსხვრევს გავრცელებულ მითს და პირდაპირ ეხება აუდიტორიის შიშს.",
        "Caption idea:",
        "პირველი შემოსავალი იწყება არა დიდი აუდიტორიით, არამედ სწორი შეთავაზებით.",
        "CTA:",
        "დაწერე 'სტარტი' და გამოგიგზავნი პირველ ნაბიჯს.",
        "",
        "IDEA 2",
        "Title:",
        "რატომ არ მოდის DM",
        "Hook:",
        "შენი Reels ნახვებს იღებს, მაგრამ DM არ მოდის?",
        "Format type:",
        "Mistake breakdown",
        "Main idea:",
        "დაანახე ერთი CTA შეცდომა და მის ნაცვლად ძლიერი ვერსია.",
        "Shot list:",
        "- სუსტი CTA",
        "- ძლიერი CTA",
        "- მაგალითი შენს ნიშაზე",
        "Why it can work:",
        "პრობლემა კონკრეტულია და გამოსავალი მარტივად გამოსაყენებელი.",
        "Caption idea:",
        "თუ CTA ბუნდოვანია, აუდიტორია შემდეგ ნაბიჯს ვერ ხედავს.",
        "CTA:",
        "დატოვე 'CTA' და ერთ ტექსტს გადაგიწერ.",
        "",
        "IDEA 3",
        "Title:",
        "ბლოგერის უხილავი შეცდომა",
        "Hook:",
        "შეიძლება კარგ კონტენტს დებდე და მაინც არ ყიდდე.",
        "Format type:",
        "POV plus tutorial",
        "Main idea:",
        "ახსენი რატომ სჭირდება personal brand-ს არა მხოლოდ რჩევები, არამედ offer logic.",
        "Shot list:",
        "1. POV ტექსტი ეკრანზე",
        "2. offer logic-ის 3 პუნქტი",
        "3. CTA კომენტარში",
        "Why it can work:",
        "კონტენტს აკავშირებს ბიზნეს მიზანთან და არა მხოლოდ reach-თან.",
        "Caption idea:",
        "ხილვადობა კარგია, მაგრამ მონეტიზაციას სტრატეგია სჭირდება.",
        "CTA:",
        "დაწერე 'offer' და გეტყვი რას შევამოწმებდი პირველ რიგში.",
    ])

    normalized = formatter.normalize_reply("reel_idea", reply)

    assert normalized["parse_status"] == "parsed"
    ideas = normalized["structured_output"]["ideas"]
    assert len(ideas) == 3
    for idea in ideas:
        assert idea["title"]
        assert idea["hook"]
        assert idea["format_type"]
        assert idea["main_idea"]
        assert isinstance(idea["shot_list"], list)
        assert idea["shot_list"]
        assert idea["why_it_can_work"]
        assert idea["caption_idea"]
        assert idea["cta"]
        assert "IDEA 2" not in idea["cta"]
        assert "IDEA 3" not in idea["cta"]
    assert ideas[0]["title"] == "მონეტიზაცია 100 გამომწერით"
    assert "გგონია" in ideas[0]["hook"]
    assert ideas[1]["shot_list"][0] == "სუსტი CTA"


def test_reel_feedback_formatter_parses_georgian_headings():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "### ✅ რას აკეთებს ეს Reel კარგად",
        "- პირადი ისტორია ქმნის ნდობას",
        "- ტონი ენერგიულია და შეესაბამება მონეტიზაციის თემას",
        "",
        "### ⚠️ რა ასუსტებს შედეგს",
        "- პირველ წამში დაძაბულობა გვიან ჩნდება",
        "- CTA ზოგადია და შემდეგ ნაბიჯს არ აზუსტებს",
        "",
        "### 📉 სად იკარგება ყურადღება / retention",
        "1. შესავალი გრძელია",
        "2. proof მომენტი გვიან ჩანს",
        "",
        "### უკეთესი hook",
        "დაიწყე პირდაპირი ფრაზით: „თუ ბლოგით ფულს ვერ გამოიმუშავებ, ეს 1 შეცდომა შეამოწმე.“",
        "",
        "### CTA-ის გაუმჯობესება",
        "დაწერე „აუდიტი“ და გამოგიგზავნი პირველ შესამოწმებელ ნაბიჯს.",
        "",
        "### საბოლოო სცენარი",
        "პირველ 2 წამში თქვი პრობლემა, შემდეგ აჩვენე მოკლე proof და ბოლოს მიეცი ერთი კონკრეტული მოქმედება.",
    ])

    normalized = formatter.normalize_reply("reel_feedback", reply)

    assert normalized["parse_status"] == "parsed"
    feedback = normalized["structured_output"]["feedback"]
    assert isinstance(feedback["what_works"], list)
    assert isinstance(feedback["what_hurts"], list)
    assert isinstance(feedback["retention_issues"], list)
    assert feedback["what_works"][0] == "პირადი ისტორია ქმნის ნდობას"
    assert "რა ასუსტებს" not in " ".join(feedback["what_works"])
    assert "ყურადღების შენარჩუნების" not in " ".join(feedback["what_hurts"])
    assert "შესავალი გრძელია" in feedback["retention_issues"][0]
    assert "თუ ბლოგით ფულს ვერ გამოიმუშავებ" in feedback["hook_improvement"]
    assert "აუდიტი" in feedback["cta_improvement"]
    assert "პირველ 2 წამში" in feedback["improved_version"]


def test_reel_script_formatter_keeps_hook_and_script_fields_clean():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "Hook:",
        "თუ შენი Reel ნახვებს იღებს, მაგრამ ლიდები არ მოდის, პრობლემა პირველ 3 წამში იწყება.",
        "",
        "Problem / angle:",
        "კონტენტი ყურადღებას იღებს, მაგრამ არ აჩვენებს რატომ უნდა მოგწეროს ადამიანმა ახლა.",
        "",
        "Scene-by-scene script:",
        "1. პირდაპირ კამერაში თქვი მთავარი ტკივილი.",
        "2. აჩვენე ერთი სუსტი CTA ეკრანზე.",
        "3. გადააკეთე ის კონკრეტულ DM მოწვევად.",
        "",
        "Voiceover:",
        "ნახვები კარგია, მაგრამ თუ CTA ბუნდოვანია, ადამიანი ვერ ხვდება რა ნაბიჯი გადადგას შემდეგ.",
        "",
        "On-screen text:",
        "- ნახვები არ ნიშნავს ლიდებს",
        "- CTA უნდა იყოს ერთი მოქმედება",
        "",
        "Caption:",
        "თუ შენი Reel ინახავს ყურადღებას, მაგრამ არ მოაქვს DM-ები, გადაამოწმე CTA.",
        "",
        "CTA:",
        "მომწერე DM-ში სიტყვა REEL და გეტყვი პირველ გამოსასწორებელ ნაბიჯს.",
    ])

    normalized = formatter.normalize_reply("reel_script", reply)

    assert normalized["parse_status"] == "parsed"
    script = normalized["structured_output"]["script"]
    assert script["hook"] == "თუ შენი Reel ნახვებს იღებს, მაგრამ ლიდები არ მოდის, პრობლემა პირველ 3 წამში იწყება."
    assert "Problem" not in script["hook"]
    assert script["problem_angle"].startswith("კონტენტი ყურადღებას იღებს")
    assert script["structure"][0] == "პირდაპირ კამერაში თქვი მთავარი ტკივილი."
    assert script["voiceover"].startswith("ნახვები კარგია")
    assert "Hook:" not in script["voiceover"]
    assert "CTA:" not in script["voiceover"]
    assert script["shot_list"] == [
        "პირდაპირ კამერაში თქვი მთავარი ტკივილი.",
        "აჩვენე ერთი სუსტი CTA ეკრანზე.",
        "გადააკეთე ის კონკრეტულ DM მოწვევად.",
    ]
    assert script["on_screen_text"] == ["ნახვები არ ნიშნავს ლიდებს", "CTA უნდა იყოს ერთი მოქმედება"]
    assert script["caption"].startswith("თუ შენი Reel ინახავს ყურადღებას")
    assert script["cta"].startswith("მომწერე DM-ში სიტყვა REEL")


def test_reel_script_embedded_json_does_not_use_full_script_as_voiceover():
    formatter = AgentResponseFormatterService()
    full_script = "\n".join([
        "Hook:",
        "თუ Reel ნახვებს იღებს, მაგრამ ლიდები არ მოდის...",
        "Problem / angle:",
        "CTA ბუნდოვანია.",
        "Scene-by-scene script:",
        "1. აჩვენე პრობლემა.",
        "CTA:",
        "მომწერე DM-ში სიტყვა REEL.",
    ])
    reply = (
        "Hook:\nთუ Reel ნახვებს იღებს, მაგრამ ლიდები არ მოდის...\n\n"
        "STRUCTURED_OUTPUT_JSON: "
        + json.dumps(
            {
                "title": "CTA clarity Reel",
                "hook": "თუ Reel ნახვებს იღებს, მაგრამ ლიდები არ მოდის...\nProblem / angle:\nCTA ბუნდოვანია.",
                "script_sections": [
                    "Problem/angle: CTA ბუნდოვანია.",
                    "1. აჩვენე პრობლემა.",
                    "2. გადააკეთე CTA.",
                    "On-screen text: ნახვები ≠ ლიდები",
                ],
                "cta": "მომწერე DM-ში სიტყვა REEL.",
                "full_script": full_script,
            },
            ensure_ascii=False,
        )
    )

    normalized = formatter.normalize_reply("reel_script", reply)

    assert normalized["parse_status"] == "parsed"
    script = normalized["structured_output"]["script"]
    assert script["hook"] == "თუ Reel ნახვებს იღებს, მაგრამ ლიდები არ მოდის..."
    assert script["voiceover"] is None
    assert script["problem_angle"] == "CTA ბუნდოვანია."
    assert script["structure"][0] == "აჩვენე პრობლემა."
    assert script["shot_list"] == ["აჩვენე პრობლემა.", "გადააკეთე CTA."]


def test_reel_script_parser_captures_all_scene_blocks_and_clean_subfields():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "Hook:",
        "თუ Reel ნახვებს იღებს, მაგრამ DM არ მოდის, CTA-ს პირველივე წამებში უნდა შეხედო.",
        "",
        "Problem / angle:",
        "ვიდეო ყურადღებას იღებს, მაგრამ არ აჩვენებს კონკრეტულ შემდეგ ნაბიჯს.",
        "",
        "Scene-by-scene script:",
        "Scene 1: პირდაპირ კამერაში თქვი პრობლემა და დაასახელე შედეგი.",
        "On-screen text (მთელი Reel განმავლობაში თანმიმდევრობით):",
        "ნახვები არ ნიშნავს ლიდებს",
        "Voiceover (რიტმი და ტონი):",
        "თუ ნახვები გაქვს, მაგრამ DM არ მოდის, პრობლემა ხშირად CTA-შია.",
        "Scene 2: აჩვენე სუსტი CTA ეკრანზე და მოკლედ ახსენი რატომ ვერ მუშაობს.",
        "On-screen text:",
        "სუსტი CTA = გაუგებარი ნაბიჯი",
        "Voiceover:",
        "როცა CTA ზოგადია, მაყურებელი ვერ ხვდება რა უნდა გააკეთოს შემდეგ.",
        "Scene 3: გადააკეთე CTA ერთ კონკრეტულ DM მოწვევად.",
        "On-screen text:",
        "ერთი CTA = ერთი ნაბიჯი",
        "Voiceover:",
        "დატოვე ერთი მოქმედება და უთხარი ზუსტად რა მოგწეროს.",
        "",
        "Caption:",
        "თუ Reel-ს ნახვები აქვს, მაგრამ ლიდები არ მოდის, დაიწყე CTA-ს გასწორებით.",
        "",
        "CTA:",
        "მომწერე DM-ში სიტყვა CTA და გაჩვენებ პირველ გამოსასწორებელ ნაბიჯს.",
    ])

    normalized = formatter.normalize_reply("reel_script", reply)

    assert normalized["parse_status"] == "parsed"
    script = normalized["structured_output"]["script"]
    assert script["hook"] == "თუ Reel ნახვებს იღებს, მაგრამ DM არ მოდის, CTA-ს პირველივე წამებში უნდა შეხედო."
    assert script["problem_angle"] == "ვიდეო ყურადღებას იღებს, მაგრამ არ აჩვენებს კონკრეტულ შემდეგ ნაბიჯს."
    assert script["structure"] == [
        "პირდაპირ კამერაში თქვი პრობლემა და დაასახელე შედეგი.",
        "აჩვენე სუსტი CTA ეკრანზე და მოკლედ ახსენი რატომ ვერ მუშაობს.",
        "გადააკეთე CTA ერთ კონკრეტულ DM მოწვევად.",
    ]
    assert script["shot_list"] == script["structure"]
    assert script["on_screen_text"] == [
        "ნახვები არ ნიშნავს ლიდებს",
        "სუსტი CTA = გაუგებარი ნაბიჯი",
        "ერთი CTA = ერთი ნაბიჯი",
    ]
    assert "Voiceover" not in " ".join(script["on_screen_text"])
    assert "რიტმი" not in " ".join(script["on_screen_text"])
    assert "ტონი" not in " ".join(script["on_screen_text"])
    assert script["voiceover"] == "\n".join([
        "თუ ნახვები გაქვს, მაგრამ DM არ მოდის, პრობლემა ხშირად CTA-შია.",
        "როცა CTA ზოგადია, მაყურებელი ვერ ხვდება რა უნდა გააკეთოს შემდეგ.",
        "დატოვე ერთი მოქმედება და უთხარი ზუსტად რა მოგწეროს.",
    ])
    assert "Caption:" not in script["voiceover"]
    assert "CTA:" not in script["voiceover"]
    assert script["caption"].startswith("თუ Reel-ს ნახვები აქვს")
    assert script["cta"].startswith("მომწერე DM-ში სიტყვა CTA")


def test_reel_feedback_formatter_separates_latest_georgian_quality_headings():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "Summary:",
        "ანალიზი ეყრდნობა ხელმისაწვდომ სიგნალებს და ანგარიშის კონტექსტს.",
        "",
        "What works / რა მუშაობს:",
        "- თემა პირდაპირ უკავშირდება აუდიტორიის ტკივილს.",
        "- proof მომენტი ნდობას ზრდის.",
        "",
        "What is weak / რა სუსტია:",
        "- დასაწყისი გვიან ამბობს კონკრეტულ შედეგს.",
        "- CTA ზედმეტად ზოგადია და შემდეგ ნაბიჯს არ აზუსტებს.",
        "",
        "Retention issues / რიტენშენის პრობლემა:",
        "1. პირველ წამში არ არის საკმარისი დაძაბულობა.",
        "2. შუა ნაწილი იმეორებს იმავე აზრს.",
        "",
        "Hook improvement / Hook-ის გაუმჯობესება:",
        "დაიწყე ასე: თუ Reel-ს ნახვები აქვს, მაგრამ DM არ მოდის, ეს CTA შეამოწმე.",
        "",
        "CTA improvement / CTA-ის გაუმჯობესება:",
        "მომწერე DM-ში სიტყვა CTA და გამოგიგზავნი პირველ გამოსასწორებელ ნაბიჯს.",
        "",
        "Improved version / გაუმჯობესებული ვერსია:",
        "პირველ 2 წამში თქვი პრობლემა, შემდეგ აჩვენე ერთი proof და ბოლოს დატოვე ერთი მკაფიო DM CTA.",
    ])

    normalized = formatter.normalize_reply("reel_feedback", reply)

    assert normalized["parse_status"] == "parsed"
    feedback = normalized["structured_output"]["feedback"]
    assert feedback["what_works"] == [
        "თემა პირდაპირ უკავშირდება აუდიტორიის ტკივილს.",
        "proof მომენტი ნდობას ზრდის.",
    ]
    assert feedback["what_hurts"] == [
        "დასაწყისი გვიან ამბობს კონკრეტულ შედეგს.",
        "CTA ზედმეტად ზოგადია და შემდეგ ნაბიჯს არ აზუსტებს.",
    ]
    assert feedback["retention_issues"] == [
        "პირველ წამში არ არის საკმარისი დაძაბულობა.",
        "შუა ნაწილი იმეორებს იმავე აზრს.",
    ]
    assert feedback["hook_improvement"].startswith("დაიწყე ასე")
    assert feedback["cta_improvement"].startswith("მომწერე DM-ში სიტყვა CTA")
    assert feedback["improved_version"].startswith("პირველ 2 წამში")
    assert "რა სუსტია" not in " ".join(feedback["what_works"])


def test_reel_feedback_formatter_keeps_english_heading_support():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "What works:",
        "- Clear topic",
        "- Strong personal proof",
        "",
        "What hurts:",
        "- The CTA is too broad",
        "",
        "Retention risks:",
        "1. The hook starts slowly",
        "2. The middle repeats the same point",
        "",
        "Better hook:",
        "If your Reel gets views but no leads, check this first.",
        "",
        "Better CTA:",
        "DM me 'REEL' and I will show you the first fix.",
        "",
        "Improved version:",
        "Start with the lead problem, show one proof point, then close with one direct CTA.",
    ])

    normalized = formatter.normalize_reply("reel_feedback", reply)

    assert normalized["parse_status"] == "parsed"
    feedback = normalized["structured_output"]["feedback"]
    assert feedback["what_works"] == ["Clear topic", "Strong personal proof"]
    assert feedback["what_hurts"] == ["The CTA is too broad"]
    assert feedback["retention_issues"][0] == "The hook starts slowly"
    assert feedback["hook_improvement"].startswith("If your Reel gets views")
    assert feedback["cta_improvement"].startswith("DM me 'REEL'")
    assert feedback["improved_version"].startswith("Start with the lead problem")


def test_profile_audit_parser_keeps_direction_sections_separate():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "Instagram profile audit",
        "",
        "What works:",
        "- Clear service promise",
        "- Proof-led Reels already match the audience pain",
        "",
        "What is weak:",
        "- Bio does not say who the offer is for",
        "- CTA is too soft for qualified inbound leads",
        "",
        "What to improve first:",
        "- Rewrite the first bio line around the client result",
        "- Replace the passive CTA with one DM action",
        "",
        "Recommended bio direction:",
        "Helping service founders turn weak content into qualified DM conversations.",
        "",
        "Content direction:",
        "- More proof-led CTA breakdowns",
        "- Fewer generic educational intros",
        "",
        "Next 3 actions:",
        "1. Rewrite the bio CTA",
        "2. Pin one proof Reel",
        "3. Post one CTA audit Reel this week",
        "",
        "Summary:",
        "The profile has a clear offer base, but conversion clarity should be tightened first.",
    ])

    normalized = formatter.normalize_reply("profile_audit", reply)

    assert normalized["parse_status"] == "parsed"
    audit = normalized["structured_output"]
    assert audit["strengths"] == [
        "Clear service promise",
        "Proof-led Reels already match the audience pain",
    ]
    assert audit["weak_points"] == [
        "Bio does not say who the offer is for",
        "CTA is too soft for qualified inbound leads",
    ]
    assert audit["quick_fixes"] == [
        "Rewrite the first bio line around the client result",
        "Replace the passive CTA with one DM action",
    ]
    assert audit["recommended_bio_direction"].startswith("Helping service founders")
    assert audit["content_direction"] == [
        "More proof-led CTA breakdowns",
        "Fewer generic educational intros",
    ]
    assert audit["priority_actions"] == [
        "Rewrite the bio CTA",
        "Pin one proof Reel",
        "Post one CTA audit Reel this week",
    ]
    assert "Recommended bio direction" not in " ".join(audit["priority_actions"])
    assert "Content direction" not in " ".join(audit["priority_actions"])
    assert audit["summary"].startswith("The profile has a clear offer")


def test_content_plan_parser_keeps_supplemental_sections_out_of_content_items():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "4-week lead generation plan",
        "",
        "Week 1:",
        "- Reel: CTA mistake breakdown",
        "- Carousel: one offer clarity checklist",
        "",
        "Week 2:",
        "- Reel: proof before pitch",
        "- Story: objection Q&A",
        "",
        "Week 3:",
        "- Reel: weak hook rewrite",
        "",
        "Week 4:",
        "- Carousel: DM conversion framework",
        "",
        "Best content mix:",
        "- 60% Reels",
        "- 25% carousels",
        "- 15% stories",
        "",
        "Hook ideas:",
        "- Your views are not the problem",
        "- This CTA quietly kills DMs",
        "",
        "CTA ideas:",
        "- DM me CTA",
        "- Send REEL for the first fix",
        "",
        "Summary:",
        "Keep the plan focused on proof, CTA clarity, and qualified DM intent.",
    ])

    normalized = formatter.normalize_reply("content_plan", reply)

    assert normalized["parse_status"] == "parsed"
    plan = normalized["structured_output"]
    assert plan["plan_title"] == "4-week lead generation plan"
    assert len(plan["content_items"]) == 6
    topics = " ".join(item["topic"] or "" for item in plan["content_items"])
    assert "60% Reels" not in topics
    assert "Your views are not the problem" not in topics
    assert "DM me CTA" not in topics
    assert plan["best_content_mix"] == ["60% Reels", "25% carousels", "15% stories"]
    assert plan["hook_ideas"] == ["Your views are not the problem", "This CTA quietly kills DMs"]
    assert plan["cta_ideas"] == ["DM me CTA", "Send REEL for the first fix"]
    assert plan["summary"].startswith("Keep the plan focused")


def test_performance_summary_parser_keeps_opportunities_and_next_actions_separate():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "What worked:",
        "- Proof-led Reels got the strongest saves",
        "- CTA audits drove more profile actions",
        "",
        "What did not work:",
        "- Generic educational posts underperformed",
        "",
        "Content patterns:",
        "- Direct hooks perform better than soft intros",
        "- DM CTAs work best after proof",
        "",
        "Best opportunities:",
        "- Turn CTA audit comments into Reels",
        "- Build a weekly proof breakdown series",
        "",
        "Next actions:",
        "1. Repeat the proof-led Reel format",
        "2. Rewrite weak CTAs into one DM action",
        "3. Test one carousel from the top Reel",
        "",
        "Summary:",
        "Double down on proof-led Reels and CTA clarity.",
    ])

    normalized = formatter.normalize_reply("performance_summary", reply)

    assert normalized["parse_status"] == "parsed"
    summary = normalized["structured_output"]
    assert summary["what_worked"] == [
        "Proof-led Reels got the strongest saves",
        "CTA audits drove more profile actions",
    ]
    assert summary["what_did_not_work"] == ["Generic educational posts underperformed"]
    assert summary["content_patterns"] == [
        "Direct hooks perform better than soft intros",
        "DM CTAs work best after proof",
    ]
    assert summary["best_opportunities"] == [
        "Turn CTA audit comments into Reels",
        "Build a weekly proof breakdown series",
    ]
    assert summary["recommended_next_moves"] == [
        "Repeat the proof-led Reel format",
        "Rewrite weak CTAs into one DM action",
        "Test one carousel from the top Reel",
    ]
    assert "Best opportunities" not in " ".join(summary["content_patterns"])
    assert "Next actions" not in " ".join(summary["content_patterns"])
    assert summary["summary"] == "Double down on proof-led Reels and CTA clarity."


def test_carousel_formatter_still_parses_after_stage4_parser_cleanup():
    formatter = AgentResponseFormatterService()
    reply = "\n".join([
        "Title:",
        "3 CTA mistakes that kill qualified DMs",
        "",
        "Slide 1:",
        "No clear offer",
        "People do not know what they get.",
        "",
        "Slide 2:",
        "Too many actions",
        "One Reel should ask for one next step.",
        "",
        "Final CTA slide:",
        "DM me CTA and I will show you the first fix.",
    ])

    normalized = formatter.normalize_reply("carousel", reply)

    assert normalized["parse_status"] == "parsed"
    carousel = normalized["structured_output"]
    assert carousel["title"] == "3 CTA mistakes that kill qualified DMs"
    assert len(carousel["slides"]) == 2
    assert carousel["slides"][0]["headline"] == "No clear offer"
    assert carousel["cta"].startswith("DM me CTA")


def test_langflow_reels_prompt_includes_high_priority_contract():
    service = LangflowService()
    prompt = service._build_input_message(
        message="Give me 3 strong Reel ideas for my current Instagram context.",
        task_type="reel_idea",
        goal="increase qualified inbound leads from Instagram",
        profile_context={
            "niche": "beauty",
            "target_audience": "women 24-35",
            "brand_voice": "direct",
            "content_focus": ["reels"],
            "strengths": ["clear offer"],
            "weak_points": ["slow hooks"],
        },
        recent_content_context={
            "top_formats": ["Reels"],
            "best_topics": ["hook logic"],
            "weak_topics": ["generic intros"],
            "best_ctas": ["DM CTA"],
            "weak_ctas": ["follow for more"],
            "notes": ["retention drops after the first second"],
        },
        recent_posts_context={"posts": [{"topic": "hook logic"}]},
        playbook_context={
            "used_system_knowledge": True,
            "matched_knowledge_domain": "reels",
            "chunks": [
                {"chunk_label": "KNOWLEDGE MODULE 1 - VIRAL IDEA MECHANICS", "text": "Use tension and simplicity."},
                {"chunk_label": "KNOWLEDGE MODULE 2 - TREND STRUCTURE", "text": "Adapt the pattern to the niche."},
            ],
        },
    )
    assert "High-priority reels methodology rules:" in prompt
    assert "STRUCTURED_OUTPUT_JSON:" in prompt
    assert '"ideas":[{"title":"...","hook":"...","format_type":"...","main_idea":"..."' in prompt
    assert "Playbook chunk 1 [KNOWLEDGE MODULE 1 - VIRAL IDEA MECHANICS]" in prompt


def test_reel_idea_route_uses_system_knowledge_and_returns_clean_structured_output(monkeypatch, tmp_path):
    monkeypatch.setenv("INTERNAL_ADMIN_KEY", INTERNAL_HEADERS["X-Internal-Admin-Key"])
    monkeypatch.setenv("USE_LANGFLOW_FOR_AGENT_CHAT", "false")
    monkeypatch.setenv("USE_LANGFLOW_SAFE_REELS_RAG", "false")

    knowledge_service = KnowledgePackService(
        data_file=tmp_path / "knowledge_packs.json",
        chunks_file=tmp_path / "knowledge_pack_chunks.json",
        storage_dir=tmp_path / "knowledge_packs_files",
    )
    history_service = GenerationHistoryService()
    history_service.data_file = tmp_path / "generation_history.json"

    monkeypatch.setattr(knowledge_pack_route, "knowledge_pack_service", knowledge_service)
    monkeypatch.setattr(agent_route, "knowledge_pack_service", knowledge_service)
    monkeypatch.setattr(agent_route, "generation_history_service", history_service)
    monkeypatch.setattr(internal_generation_debug_route, "generation_history_service", history_service)
    monkeypatch.setattr(agent_route.langflow_service, "run_agent", lambda **kwargs: (_ for _ in ()).throw(AssertionError("Langflow should not be called when USE_LANGFLOW_FOR_AGENT_CHAT=false")))

    captured_calls: list[dict] = []

    monkeypatch.setattr(agent_route.billing_service, "enforce_agent_access", lambda user_id, task_type: {"current_plan": "pro"})
    monkeypatch.setattr(agent_route.billing_service, "increment_generation_usage", lambda user_id, task_type=None: None)
    monkeypatch.setattr(agent_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id=None: account_id or "acct-1")
    monkeypatch.setattr(agent_route.connected_accounts_service, "find_user_id_by_account_id", lambda account_id: "user-1")
    monkeypatch.setattr(agent_route.instagram_context_sync_service, "get_context_freshness", lambda account_id: {
        "context_was_fresh": True,
        "last_synced_at": "2026-04-27T11:45:00+00:00",
        "stale_reasons": [],
        "has_complete_context": True,
    })
    monkeypatch.setattr(agent_route.profile_context_service, "get_context", lambda account_id: {
        "brand_name": "Lead Lab",
        "niche": "service marketing",
        "target_audience": "service founders",
        "brand_voice": "direct",
        "bio": "We turn weak DMs into qualified leads",
        "content_focus": ["reels", "proof"],
        "strengths": ["clear offer"],
        "weak_points": ["slow CTA"],
    })
    monkeypatch.setattr(agent_route.recent_posts_context_service, "get_context", lambda account_id: {
        "posts": [
            {"post_id": "p1", "content_type": "REEL", "topic": "CTA mistakes", "caption": "CTA fixes", "views": 1000, "likes": 98, "comments": 12, "saves": 21},
            {"post_id": "p2", "content_type": "REEL", "topic": "hooks", "caption": "Hook logic", "views": 900, "likes": 86, "comments": 9, "saves": 17},
        ]
    })
    monkeypatch.setattr(agent_route.recent_content_context_service, "get_context", lambda account_id: {
        "top_formats": ["Reels"],
        "best_topics": ["CTA mistakes", "hook logic"],
        "weak_topics": ["generic educational intros"],
        "best_ctas": ["DM CTA"],
        "weak_ctas": ["follow for more"],
        "notes": ["retention drops after the first second", "proof-led content converts better"],
    })
    structured_payload = json.loads(_build_reel_idea_reply().split("STRUCTURED_OUTPUT_JSON: ", 1)[1])

    def fake_run_agent(**kwargs):
        captured_calls.append(kwargs)
        return {
            "reply": _build_reel_idea_reply().split("STRUCTURED_OUTPUT_JSON:", 1)[0].rstrip(),
            "account_id": kwargs.get("account_id"),
            "structured_output": structured_payload,
            "parse_status": "parsed",
            "model_provider": "openai",
            "model_name": "gpt-5.2",
            "used_langflow": False,
            "prompt_token_estimate": 1280,
            "retry_count": 0,
            "rate_limited": False,
            "prompt_section_names": [
                "base_system_instruction",
                "task_instruction",
                "reels_high_priority_instruction",
                "internal_strategy_context",
                "user_request",
            ],
        }

    monkeypatch.setattr(agent_route.llm_service, "run_agent", fake_run_agent)

    client = TestClient(app)
    _upload_internal_reels_pack(client)

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "Give me 3 strong Reel ideas for my current Instagram context.",
            "task_type": "reel_idea",
            "user_id": "user-1",
            "goal": "increase qualified inbound leads from Instagram",
            "auto_sync": True,
        },
    )
    assert response.status_code == 200, response.text

    response_payload = response.json()
    assert response_payload["parse_status"] == "parsed"
    assert "STRUCTURED_OUTPUT_JSON:" not in response_payload["reply"]
    assert "Format type:" in response_payload["reply"]
    assert "Why it can work:" in response_payload["reply"]
    assert "Use tension and simplicity." not in response_payload["reply"]
    assert "Adapt the pattern to the niche" not in response_payload["reply"]

    ideas = response_payload["structured_output"]["ideas"]
    assert len(ideas) == 3
    for idea in ideas:
        assert idea["title"]
        assert idea["format_type"]
        assert idea["main_idea"]
        assert idea["why_it_can_work"]
        assert idea["cta"]
        assert "###" not in idea["cta"]
        assert "Reel 2" not in idea["cta"]
        assert "Reel 3" not in idea["cta"]
        assert not idea["title"].startswith("Here are 3 strong")

    captured_call = captured_calls[-1]
    playbook_context = captured_call["playbook_context"]
    assert playbook_context["used_system_knowledge"] is True
    assert playbook_context["matched_knowledge_domain"] == "reels"
    assert playbook_context["retrieved_chunk_count"] > 0
    assert playbook_context["retrieved_chunk_titles"]

    debug_response = client.get(
        "/api/v1/internal/generation-debug/latest",
        headers=INTERNAL_HEADERS,
        params={"user_id": "user-1", "task_type": "reel_idea"},
    )
    assert debug_response.status_code == 200, debug_response.text
    debug_payload = debug_response.json()
    assert debug_payload["used_system_knowledge"] is True
    assert debug_payload["matched_knowledge_domain"] == "reels"
    assert debug_payload["retrieved_chunk_count"] > 0
    assert debug_payload["retrieved_chunk_titles"]
    assert debug_payload["used_langflow"] is False
    assert debug_payload["model_provider"] == "openai"
    assert debug_payload["model_name"] == "gpt-5.2"
    assert "internal_strategy_context" in debug_payload["prompt_section_names"]
    assert debug_payload["prompt_token_estimate"] == 1280
    assert debug_payload["retry_count"] == 0
    assert debug_payload["rate_limited"] is False
    assert debug_payload["parse_status"] == "parsed"
    assert "chunks" not in debug_payload

    history_item = history_service.get_latest_item(user_id="user-1", task_type="reel_idea")
    assert history_item is not None
    serialized_history = json.dumps(history_item, ensure_ascii=False)
    assert "Use tension and simplicity." not in serialized_history
    assert "Adapt the pattern to the niche" not in serialized_history
    assert history_item["used_langflow"] is False
    assert history_item["model_provider"] == "openai"
    assert history_item["prompt_token_estimate"] == 1280
    assert history_item["retry_count"] == 0
    assert history_item["rate_limited"] is False
