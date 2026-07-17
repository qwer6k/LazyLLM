import os
from contextlib import contextmanager
from pathlib import Path
from typing import List

import pytest
from pydantic import BaseModel

import lazyllm
import lazyllm.tools.fs.supplier.feishu  # noqa: F401 — triggers config.add registrations
from lazyllm.tools.fs.client import dynamic_fs_config
from lazyllm.tools.writer.tools.base import WriterToolBase
from lazyllm.tools.writer.data_models.context import DocumentSummary, WritingContext
from lazyllm.tools.writer.data_models.docir import DocIR
from lazyllm.tools.writer.data_models.quality import AuditResult, ReviewReport
from lazyllm.tools.writer.data_models.revision import LocateResult, ModifyPlan, PatchResult, PatchSet
from lazyllm.tools.writer.data_models.task import InputResource, Selection, TargetDocument, WritingTask
from lazyllm.tools.writer.data_models.writing import (
    DraftBlock,
    DraftDocument,
    DraftSection,
    SectionInstructionList,
    WritingOutline,
    WritingOutput,
)
from lazyllm.tools.writer.workflow.naive_writer_workflow import NaiveWriterWorkflow
from lazyllm.tools.writer.utils import load_artifact_json
from ...utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineChatModuleBase.py'
FEISHU_WRITE_TARGET = os.environ.get('FEISHU_WRITE_TARGET')
WRITER_BASE_PATH = 'lazyllm/tools/writer/tools/base.py'


@contextmanager
def _feishu_auth_context():
    '''Set up dynamic_fs_auth for Feishu when environment is configured.'''
    token = _acquire_feishu_token()
    if token:
        with dynamic_fs_config({'feishu': token}):
            yield token
    else:
        yield None


def _acquire_feishu_token():
    try:
        app_id = lazyllm.config['feishu_app_id'] or os.environ.get('FEISHU_APP_ID', '')
        app_secret = lazyllm.config['feishu_app_secret'] or os.environ.get('FEISHU_APP_SECRET', '')
    except KeyError:
        app_id = os.environ.get('FEISHU_APP_ID', '')
        app_secret = os.environ.get('FEISHU_APP_SECRET', '')
    if not app_id or not app_secret:
        return None
    import requests as _req
    resp = _req.post(
        'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
        json={'app_id': app_id, 'app_secret': app_secret},
        headers={'Content-Type': 'application/json; charset=utf-8'},
        timeout=10,
    )
    data = resp.json()
    if data.get('code', -1) != 0:
        return None
    return data.get('tenant_access_token', '')
QWEN_MODEL = 'qwen-turbo'


class WriterStructuredProbe(BaseModel):
    title: str
    section_count: int
    keywords: List[str]


@pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'), WRITER_BASE_PATH)
def test_writer_call_llm_structured_with_qwen():
    llm = lazyllm.OnlineChatModule(
        source='qwen',
        model=QWEN_MODEL,
        api_key=get_api_key('qwen'),
        stream=False,
    )
    tool = WriterToolBase(llm=llm)

    result = tool._call_llm_structured(
        (
            'Generate a compact JSON object for testing WriterToolBase structured LLM output. '
            'Use title \'Writer Pipeline Structured Output Test\', section_count 3, '
            'and include the keywords planning, drafting, and review.'
        ),
        WriterStructuredProbe,
    )

    assert isinstance(result, WriterStructuredProbe)
    assert result.title
    assert result.section_count == 3
    assert {'planning', 'drafting', 'review'}.issubset(set(result.keywords))


# ============================================================================
# NaiveWriterWorkflow.write() E2E
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_stage(stages: dict, key: str, model_class=None):
    entry = stages.get(key) or {}
    if not isinstance(entry, dict):
        return None
    path = (
        entry.get('metadata', {}).get('artifact_paths', {}).get(key)
        or entry.get('artifact_path', '')
    )
    if not path:
        return None
    return load_artifact_json(path, model_class)


def test_write_workflow_e2e():
    '''Run NaiveWriterWorkflow.write() end-to-end and verify every stage's artifact.'''
    llm = lazyllm.OnlineChatModule(
        source='qwen', model=QWEN_MODEL,
        api_key=get_api_key('qwen'), stream=False,
    )
    store = str(REPO_ROOT / 'tests' / 'charge_tests' / 'artifacts' / 'write_workflow_e2e')
    wf = NaiveWriterWorkflow(llm=llm, artifact_store=store)

    inputs = [
        InputResource(
            resource_type='text', resource_id='r1', title='需求规格',
            inline_text=(
                'The product is an AI coding assistant that supports Python, JavaScript, '
                'TypeScript, Go, Java, and Rust. Backend uses microservices architecture '
                'with Python and Go. Must support on-premises deployment and SaaS multi-tenant. '
                'Frontend is a VS Code extension and JetBrains plugin.'
            ),
        ),
        InputResource(
            resource_type='text', resource_id='r2', title='DeepSeek V4 技术报告',
            inline_text=(
                'DeepSeek V4 is a large language model with 685B parameters, '
                'using Mixture-of-Experts architecture. It supports long-context '
                'of 1M tokens and achieves competitive performance on coding benchmarks.'
            ),
        ),
        InputResource(
            resource_type='text', resource_id='r3', title='市场数据',
            inline_text=(
                'The AI coding assistant market reached $3.2 billion in 2024. '
                'GitHub Copilot has over 1.8 million paying users. User willingness '
                'to pay is concentrated on accuracy and latency.'
            ),
        ),
    ]
    task = WritingTask(
        task_id='wf-e2e',
        query=(
            'Write a technical overview for an AI-powered coding assistant product. '
            'Cover system architecture, supported languages, deployment model, and security. '
            'Include a comparison table of features vs. competitors.'
        ),
        task_type='write',
        inputs=inputs,
        target_document=TargetDocument(uri=FEISHU_WRITE_TARGET) if FEISHU_WRITE_TARGET else None,
    )

    with _feishu_auth_context():
        result = wf.write(
            task=task.model_dump(),
        )
    stages = result.get('stage_results') or {}
    assert stages, 'stage_results must not be empty'

    # --- Step 1: resource_profiles ---
    profiles = _load_stage(stages, 'resource_profiles')
    assert isinstance(profiles, list), f'Expected list, got {type(profiles)}'
    assert len(profiles) >= 3, f'Expected >=3 profiles, got {len(profiles)}'
    for p_dict in profiles:
        # loaded as dict when model_class=None; validate expected keys
        assert isinstance(p_dict.get('resource_id'), str)
        assert p_dict.get('resource_role') in ('spec', 'background', 'example')
        assert isinstance(p_dict.get('key_facts'), list)

    # --- Step 2: writing_context ---
    ctx = _load_stage(stages, 'writing_context', WritingContext)
    assert ctx is not None
    assert ctx.context_id == 'wf-e2e'
    assert len(ctx.facts) >= 1
    assert ctx.document_summary is not None
    assert len(ctx.document_summary.key_points) >= 2

    # --- Step 3: outline ---
    outline = _load_stage(stages, 'outline', WritingOutline)
    assert outline is not None
    assert len(outline.nodes) >= 1

    # --- Step 4: section_instructions ---
    instructions = _load_stage(stages, 'section_instructions', SectionInstructionList)
    assert instructions is not None
    assert len(instructions.instructions) >= 1
    assert instructions.instructions[0].section_title

    # --- Step 5: draft_section ---
    section = _load_stage(stages, 'draft_section', DraftSection)
    assert section is not None
    assert section.title
    assert len(section.blocks) >= 2
    assert section.blocks[0].content, 'Block 0 has empty content'

    # --- Step 6: section_review ---
    review = _load_stage(stages, 'section_review', ReviewReport)
    assert review is not None
    assert isinstance(review.result.is_passed, bool)
    assert 0 <= review.result.score <= 100

    # --- Step 7: writing_context (updated) ---
    ctx2 = _load_stage(stages, 'writing_context', WritingContext)
    assert ctx2 is not None
    assert ctx2.document_summary.summary
    assert len(ctx2.meta.get('context_updates', [])) >= 1

    # --- Step 8: draft_document ---
    doc = _load_stage(stages, 'draft_document', DraftDocument)
    assert doc is not None
    assert doc.title
    assert len(doc.sections) >= 1

    # --- Step 9: writing_output ---
    output = _load_stage(stages, 'writing_output', WritingOutput)
    assert output is not None
    assert output.title
    assert len(output.content) >= 100
    assert output.output_format == 'markdown'

    # --- Step 10: output_review ---
    out_review = _load_stage(stages, 'draft_document_review', ReviewReport)
    assert out_review is not None
    assert isinstance(out_review.result.is_passed, bool)
    assert 0 <= out_review.result.score <= 100

    # --- primary_result ---
    primary = result.get('primary_result') or {}
    primary_path = primary.get('artifact_path') if isinstance(primary, dict) else ''
    assert primary_path

    # --- write_result (only when feishu target is configured) ---
    if FEISHU_WRITE_TARGET:
        write_result = _load_stage(stages, 'write_result')
        assert write_result, 'write_result must not be empty when target is set'
        assert write_result.get('adapter') == 'feishu', f'Expected feishu adapter: {write_result}'
        assert write_result.get('locator') == FEISHU_WRITE_TARGET
        assert write_result.get('doc_id'), (
            f'Expected non-empty doc_id, got: {write_result}'
        )


# ============================================================================
# NaiveWriterWorkflow.revise() E2E
# ============================================================================


def test_revise_workflow_e2e():
    '''End-to-end verify NaiveWriterWorkflow.revise() against a deeply nested draft
    with cross-level blocks, mixed modify types (replace/delete/insert),
    section-level exclusion, and per-block preservation.'''
    llm = lazyllm.OnlineChatModule(
        source='qwen', model=QWEN_MODEL,
        api_key=get_api_key('qwen'), stream=False,
    )
    store = str(REPO_ROOT / 'tests' / 'charge_tests' / 'artifacts' / 'revise_workflow_e2e')
    wf = NaiveWriterWorkflow(llm=llm, artifact_store=store)

    section_a = DraftSection(
        section_id='sec-intro',
        title='Introduction',
        blocks=[
            DraftBlock(block_id='int-title', content='LazyCoder Product Guide'),
            DraftBlock(block_id='int-summary',
                       content='Comprehensive overview of the AI-powered coding assistant.'),
            DraftBlock(block_id='int-scope',
                       content='This document covers features, architecture, and deployment.'),
        ],
    )
    section_b = DraftSection(
        section_id='sec-arch',
        title='Architecture',
        sub_sections=[
            DraftSection(
                section_id='arch-backend', title='Backend',
                blocks=[
                    DraftBlock(block_id='be-msa',
                               content='Microservices architecture with Python and Go.'),
                    DraftBlock(block_id='be-db',
                               content='PostgreSQL for structured data, Redis for caching.'),
                ],
            ),
            DraftSection(
                section_id='arch-frontend', title='Frontend',
                sub_sections=[
                    DraftSection(
                        section_id='arch-fe-vscode', title='VS Code Extension',
                        blocks=[
                            DraftBlock(block_id='vsc-install',
                                       content='Install via marketplace or .vsix file.'),
                            DraftBlock(block_id='vsc-lsp',
                                       content='LSP integration for IntelliSense and diagnostics.'),
                        ],
                    ),
                    DraftSection(
                        section_id='arch-fe-jetbrains', title='JetBrains Plugin',
                        blocks=[
                            DraftBlock(block_id='jb-install',
                                       content='Install from JetBrains Marketplace.'),
                            DraftBlock(block_id='jb-perf',
                                       content='Performance lags on large projects with >10k files.'),
                        ],
                    ),
                ],
            ),
        ],
    )
    section_c = DraftSection(
        section_id='sec-pricing',
        title='Pricing & Plans',
        blocks=[
            DraftBlock(block_id='price-intro',
                       content='We offer flexible plans for teams of all sizes.'),
            DraftBlock(block_id='price-free',
                       content='**Free**: 100 completions/day, community support.'),
            DraftBlock(block_id='price-pro',
                       content='**Pro**: unlimited completions, priority support. $20/month.'),
            DraftBlock(block_id='price-enterprise',
                       content='**Enterprise**: on-premises, SSO, audit logs. Custom pricing.'),
            DraftBlock(block_id='price-note',
                       content='All plans include a 14-day free trial with no credit card required.'),
        ],
    )
    draft = DraftDocument(
        draft_id='draft-1',
        title='LazyCoder Product Overview',
        sections=[section_a, section_b, section_c],
    )
    context = WritingContext(
        context_id='revise-e2e',
        doc_id='draft-1',
        document_summary=DocumentSummary(summary='LazyCoder Product Overview', key_points=[]),
    )

    result = wf.revise(
        task=WritingTask(
            task_id='revise-e2e',
            task_type='revise',
            constraints={
                'exclude_section_ids': ['sec-intro'],
                'preserve_block_ids': ['vsc-install', 'price-note'],
            },
            selection=Selection(
                scope='selection',
                block_ids=[
                    'be-msa',           # L2, arch/backend
                    'vsc-lsp',          # L4, arch/frontend/vscode
                    'jb-perf',          # L4, arch/frontend/jetbrains
                    'price-pro',        # L1, pricing
                    'price-enterprise',  # L1, pricing (delete)
                    'price-free',       # L1, pricing (anchor for insert)
                ],
            ),
            query=(
                'Multiple changes across the document:\n'
                '1. Update be-msa: add that the system also uses Rust for '
                'performance-critical paths.\n'
                '2. Update vsc-lsp: mention support for inlay hints and code actions.\n'
                '3. Fix jb-perf: update to say "performance has been significantly '
                'improved in v2.1"\n'
                '4. Update price-pro: change to $25/month.\n'
                '5. Delete price-enterprise entirely (product decision).\n'
                '6. Insert a new Team plan block after price-free: '
                '"**Team**: 500 completions/day per seat, Slack support. $12/seat/month."\n'
                '7. Do NOT modify any blocks in the Introduction section.\n'
                '8. Do NOT modify vsc-install or price-note.\n'
            ),
        ).model_dump(),
        document=draft,
        context=context,
    )
    stages = result.get('stage_results') or {}
    assert stages, 'revise() stage_results must not be empty.'

    # --- locate ---
    locate = _load_stage(stages, 'locate_result', LocateResult)
    assert locate.target_block_ids, 'locate must select at least one block.'
    allowed = {'be-msa', 'vsc-lsp', 'jb-perf', 'price-pro',
               'price-enterprise', 'price-free'}
    assert set(locate.target_block_ids) <= allowed, (
        f'locate must only pick blocks within selection, got {locate.target_block_ids}'
    )
    for bid in locate.target_block_ids:
        assert locate.target_reasons.get(bid, '').strip(), f'missing reason for selected block {bid}.'

    # --- modify_plan ---
    plan = _load_stage(stages, 'modify_plan', ModifyPlan)
    assert {i.target_block_id for i in plan.instructions} == set(locate.target_block_ids)
    modify_types = {i.modify_type for i in plan.instructions}
    assert 'replace' in modify_types, f'Expected replace in plan, got {modify_types}'
    for instr in plan.instructions:
        assert instr.modify_type in {'insert', 'replace', 'delete'}
        assert instr.instruction.strip()

    # --- patch_set ---
    patch = _load_stage(stages, 'patch_set', PatchSet)
    assert len(patch.hunks) >= 5, f'Expected >=5 hunks, got {len(patch.hunks)}'
    original_text_by_id = {}
    for s in [section_a, section_b, section_c]:
        for b in _iter_draft_blocks(s):
            original_text_by_id[b.block_id] = b.content

    for hunk in patch.hunks:
        assert hunk.anchor is not None and hunk.anchor.block_id == hunk.target_block_id
    assert {h.target_block_id for h in patch.hunks} <= allowed

    # --- per-operation assertions ---
    # replace (L2 — backend)
    be_hunk = next((h for h in patch.hunks if h.target_block_id == 'be-msa'), None)
    assert be_hunk and be_hunk.modify_type == 'replace', 'be-msa must be replaced'
    assert 'Rust' in be_hunk.new_text, f'Expected Rust in be-msa: {be_hunk.new_text[:100]}'

    # replace (L4 — vscode)
    lsp_hunk = next((h for h in patch.hunks if h.target_block_id == 'vsc-lsp'), None)
    assert lsp_hunk and lsp_hunk.modify_type == 'replace', 'vsc-lsp must be replaced'
    assert 'inlay' in lsp_hunk.new_text.lower(), (
        f'Expected inlay hints in vsc-lsp: {lsp_hunk.new_text[:100]}'
    )

    # replace (L4 — jetbrains perf fix)
    perf_hunk = next((h for h in patch.hunks if h.target_block_id == 'jb-perf'), None)
    assert perf_hunk and perf_hunk.modify_type == 'replace', 'jb-perf must be replaced'
    perf_lower = perf_hunk.new_text.lower()
    assert 'improved' in perf_lower or 'v2.1' in perf_lower, (
        f'Expected perf improvement in jb-perf: {perf_hunk.new_text[:100]}'
    )

    # delete
    del_hunk = next((h for h in patch.hunks if h.target_block_id == 'price-enterprise'), None)
    assert del_hunk and del_hunk.modify_type == 'delete', (
        f'price-enterprise must be deleted, got {del_hunk.modify_type if del_hunk else "missing"}'
    )

    # insert
    ins_hunk = next((h for h in patch.hunks if h.modify_type == 'insert'), None)
    assert ins_hunk, 'Expected at least one insert hunk (Team plan)'
    assert 'Team' in ins_hunk.new_text, f'Expected Team plan in insert: {ins_hunk.new_text[:100]}'

    # --- patch_review ---
    review = _load_stage(stages, 'patch_review', AuditResult)
    assert review is not None
    assert isinstance(review.is_passed, bool)
    assert 0 <= review.score <= 100

    # --- apply_patch ---
    patch_result = _load_stage(stages, 'patch_result', PatchResult)
    assert patch_result is not None and patch_result.success
    assert not patch_result.failed_hunks

    # --- revised_doc_ir ---
    revised_ir = load_artifact_json(stages['revised_doc_ir'], DocIR)
    revised_text_by_id = {b.block_id: b.text for b in revised_ir.blocks}

    # modified blocks changed
    for bid in ['be-msa', 'vsc-lsp', 'jb-perf', 'price-pro']:
        assert revised_text_by_id.get(bid) != original_text_by_id[bid], (
            f'{bid} should have changed'
        )

    # excluded section (sec-intro) — completely unchanged
    for bid in ['int-title', 'int-summary', 'int-scope']:
        assert revised_text_by_id[bid] == original_text_by_id[bid], (
            f'{bid} in excluded section should not change'
        )

    # preserved blocks — unchanged
    for bid in ['vsc-install', 'price-note']:
        assert revised_text_by_id[bid] == original_text_by_id[bid], (
            f'{bid} (preserved) should not change'
        )

    # other unselected blocks unchanged
    for bid in ['be-db', 'jb-install', 'price-intro']:
        assert revised_text_by_id[bid] == original_text_by_id[bid], (
            f'{bid} (unselected) should not change'
        )

    # deleted block absent
    assert 'price-enterprise' not in revised_text_by_id or \
        not revised_text_by_id['price-enterprise'], (
        'price-enterprise should not appear in revised DocIR'
    )

    # --- rebuild + writing_output ---
    revised_draft = _load_stage(stages, 'revised_draft', DraftDocument)
    assert revised_draft is not None and revised_draft.sections and revised_draft.title

    revised_context = _load_stage(stages, 'writing_context', WritingContext)
    assert revised_context is not None and revised_context.draft_document is not None

    output = _load_stage(stages, 'writing_output', WritingOutput)
    assert output is not None and output.output_format == 'markdown'
    assert len(output.content) >= 100
    assert 'rust' in output.content.lower(), 'Rust must appear in the final output.'
    content_lower = output.content.lower()
    assert 'inlay' in content_lower, 'Inlay hints must appear in the final output.'
    assert 'team' in content_lower, 'Team plan must appear in the final output.'
    assert 'enterprise' not in content_lower, 'Enterprise should not appear in the final output.'

    primary = result.get('primary_result') or {}
    assert primary.get('artifact_path'), 'primary_result must carry an artifact_path.'


def _iter_draft_blocks(section):
    '''Yield all DraftBlocks from a DraftSection recursively.'''
    for b in section.blocks:
        yield b
    for sub in section.sub_sections:
        yield from _iter_draft_blocks(sub)
