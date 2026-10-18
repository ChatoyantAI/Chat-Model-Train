"""Build an offline CoSER trajectory review from immutable exported results.

No API calls. Inputs remain unchanged; JSON is embedded safely in one HTML file.
Usage: python scripts/evaluation/build_storyline_comparison.py
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DIMS = ['Storyline Consistency', 'Anthropomorphism', 'Character Fidelity', 'Storyline Quality']

# Review notes are deliberately separate from the judge's verbatim critiques.
NOTES = {
    196: [
        ('full', 'No, they were killed', '可直接核对的矛盾', '前一轮问“他们被杀了吗？”，接着回答“No, they were killed”。否定词与后半句直接冲突；后续也没有解释。这个问题不依赖是否复现原著。'),
        ('full', 'Winterfell massacre', '突然引入事件', '对“到达长城时发生了什么”的提问，突然转到“别怪我造成临冬城屠杀”。给定场景没有铺垫这件事，后续对话也没有把它讲清楚。'),
        ('full', 'pie crust', '语义衔接松散', '从对方说肚子里装满派，接到“我们很快会变成派和派皮”。可能想表达威胁或玩笑，但上下文缺少让这句成立的具体关系。'),
        ('it', 'Watchman', 'Base 也不是无错', '这句由 Jon Snow 生成，却把对方称作 Watchman；Jon 自己才是守夜人。原评分输入不少轮次没有说话者名字，这类归属错误可能被掩盖。高分不等于文本没有问题。'),
    ],
    127: [
        ('full', "Patrick's chair", '角色与物件归属', '前文 Patrick 已站起来，这里却推起“Patrick 的椅子”；轮椅使用者是 Susannah。请同时查看场景、角色资料和前后轮次，判断有没有合理解释。'),
        ('full', 'wyverns', '额外事件与环境反馈', '对话引入飞龙，之后环境模拟器继续扩大危机。灰色 Environment 段落由环境模型生成，不能把整个失控过程都归给被测模型。'),
        (None, '', '参考信息不等于模型输入', 'Judge 会对照原故事的谷仓计划扣分，但这次 actor 请求没有得到该计划所在的 inner-state 字段。因此，“没按原著走”与“把轮椅归属写错”是两类问题，不能混作同一个训练缺陷。'),
    ],
    46: [
        ('full', 'dutiful', '微调后更好的例子', '关系冲突更具体：她追问是真正在乎还是只尽未婚夫的责任，接着试探接纳、表达不安，再提出离开。不是所有场景都退步。'),
        (None, '', '改善有多大仍需谨慎', 'Base 多次重复风暴、光、黎明的安慰比喻。微调版也仍有病句；两次评分的优势幅度明显不同。请结合全文，不只看单次分差。'),
        (None, '', 'Judge 扣分也需要检查', '本次 Base 的剧情质量评语中出现“No factual contradictions detected”却给 severity 1。下方保留原始评语和计分过程，未替 Judge 自动修正。'),
    ],
    93: [
        ('full', 'save her life from grief', '高分反例：生死冲突', '先说“She is dead”，后来又说“But you can save her life from grief”。文本没有解释如何挽救已经死去的人；这段仍然获得很高的剧情质量分。'),
        ('full', 'name of the man', '指代与记忆混乱', '开头把爱人称作 man / his，后面又用 her；Morrel 还说不知道她的名字。阅读场景背景后，再判断这些说法是否符合人物处境。'),
        (None, '', '为什么还能拿高分', '分数由 Judge 列出的问题及 severity 决定，再加每个角色回合 1.5 分，并封顶 100。少列问题、给低 severity、长对话的回合加分，都可能让明显有问题的文本拿高分。HER 这次输入完全没变，重评分变化体现了评分波动。'),
    ],
    18: [
        ('full', 'woman I fell in love', '人物性别指代错位', 'Faile 在心里用“我爱上的那个女人”指 Perrin。角色心理也在保存的 Judge 视图中，并没有统一被删除。'),
        (None, '', '不要把不同版本混在一起', '先前 LoRA epoch 2 的这个场景出现过“We’re here”循环；当前展示的是全参数 epoch 1，不能把旧版本的复读片段当成这一版的输出。'),
    ],
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def labelled(simulation):
    parts = []
    for turn in simulation:
        role, content = turn['role'], turn['content']
        prefix = r'^\s*(?:\*\*)?' + re.escape(role) + r'(?:\*\*)?\s*[:：](?:\*\*)?'
        if role != 'Environment' and not re.match(prefix, content):
            content = role + ': ' + content
        parts.append(content.strip('\n'))
    return '\n\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / '.local/storyline-audit-20260929')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/evaluation/GEMMA_E2B_STORYLINE_COMPARISON_20260929.html')
    args = parser.parse_args()
    results = read(args.input_dir / 'results.json')
    rejudge = read(args.input_dir / 'label-rejudge-results.json')
    dataset_path = ROOT / 'vendor/benchmarks/CoSER/data/test/test_set.json'
    dataset = read(dataset_path)
    new = {model: {(row['index'], row['dimension']): row for row in rejudge['rows'][group]
                   if row.get('status') == 'completed'} for model, group in [('it', '0'), ('full', '1')]}
    ids = sorted(set(map(int, results['it']['results'])) & set(map(int, results['full']['results'])))
    assert ids == rejudge['manifest']['case_ids']
    cases = []
    for index in ids:
        scene = {'id': index, 'book': results['it']['results'][str(index)]['book'],
                 'reference': dataset[index], 'models': {}, 'notes': []}
        for model in ['it', 'full']:
            r = results[model]['results'][str(index)]
            assert len(r['cases']) == 1
            c = next(iter(r['cases'].values()))
            assert c['simulation_str'] == '\n\n'.join(t['content'].strip('\n') for t in c['simulation'])
            scene['models'][model] = {
                'simulation': c['simulation'], 'oldInput': c['simulation_str'], 'newInput': labelled(c['simulation']),
                'old': {dim: {'score': r['scores'][dim], 'critique': c['critique'][dim]} for dim in DIMS},
                'new': {dim: new[model].get((index, dim)) for dim in DIMS},
                'actorTurns': sum(t['role'] != 'Environment' for t in c['simulation']),
            }
        for model, needle, title, text in NOTES.get(index, []):
            turn = None
            if model:
                matches = [n for n, t in enumerate(scene['models'][model]['simulation']) if needle.casefold() in t['content'].casefold()]
                if not matches:
                    raise ValueError(f'Annotation quote missing: case={index}, model={model}, quote={needle}')
                turn = matches[0]
            scene['notes'].append({'model': model, 'turn': turn, 'title': title, 'text': text, 'quote': needle})
        cases.append(scene)
    history = []
    for key, title in [('it_repeat2', 'IT · 9/26 重跑'), ('it', 'IT · 9/27 本次对照'), ('lora1', 'LoRA · epoch 1'), ('lora2', 'LoRA · epoch 2'), ('full', '全参数 · epoch 1')]:
        rr = results[key]['results']
        history.append({'name': title, 'source': results[key]['name'],
                        'scores': {dim: sum(rr[str(i)]['scores'][dim] for i in ids) / len(ids) for dim in DIMS}})
    paths = [args.input_dir / 'results.json', args.input_dir / 'label-rejudge-results.json', dataset_path]
    data = {'cases': cases, 'dimensions': DIMS, 'history': history, 'manifest': rejudge['manifest'],
            'snapshot': datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8'),
            'provenance': [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            'completedJudgments': sum(len(v) for v in new.values()),
            'failures': [{'model': model, 'index': row['index'], 'dimension': row['dimension'], 'error': row.get('error', 'unknown')}
                         for model, group in [('it', '0'), ('full', '1')] for row in rejudge['rows'][group] if row.get('status') == 'failed'],
            'failedJudgments': sum(row.get('status') == 'failed' for rows in rejudge['rows'].values() for row in rows)}
    template = Path(__file__).with_name('storyline_comparison.html').read_text(encoding='utf-8')
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace('__EMBEDDED_DATA__', payload), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'cases': len(cases), 'completed_judgments': data['completedJudgments'], 'bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
