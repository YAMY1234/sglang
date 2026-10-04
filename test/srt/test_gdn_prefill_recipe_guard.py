"""Startup log contracts and production init ordering; CPU, no torch import."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
NAME = 'sglang.srt.mem_cache.gdn_prefill_recipe_guard'
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(NAME, ROOT/'python/sglang/srt/mem_cache/gdn_prefill_recipe_guard.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
g = sys.modules[NAME]


def runner(role='prefill', *, rank=8, shallow=False):
    owner = NS(fullstack={'gdn_rank':rank}, pd_shallow_role='prefill' if shallow else None)
    return NS(model=owner, server_args=NS(disaggregation_mode=role),
              req_to_token_pool=NS(factored_gdn_pool=NS(cfg=NS(r=rank))))


def messages(records):
    return [json.loads(r.getMessage().split(' ',4)[4]) for r in records]


def method(path, cls_name, method_name, scope):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls_name)
    func=copy.deepcopy(next(n for n in cls.body if getattr(n,'name','')==method_name))
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,func],type_ignores=[])),str(path),'exec'),scope)
    return scope[method_name]


class RecipeGuardTest(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{},clear=True);self.env.start();self.addCleanup(self.env.stop)

    def report(self,r):
        with self.assertLogs(g.logger,level='INFO') as cap:g.report_startup(r)
        return cap.records,messages(cap.records)

    def test_factor_enabled_without_install_warns_and_names_missing_keys(self):
        records,rows=self.report(runner())
        self.assertEqual([r.levelname for r in records],['WARNING','WARNING'])
        exact,fulln=rows
        self.assertEqual(exact['status'],'not-installed')
        self.assertEqual(exact['missing_switches'],[g.EXACT+'=1',g.COMMIT+'=1',g.FACTOR_ONLY+'=1'])
        self.assertNotIn(g.FULL_N+'=1',fulln['missing_switches']) # existing default is 1
        self.assertEqual(exact['switches'][g.EXACT],'<unset>')

    def test_success_has_info_receipts_and_no_warning(self):
        r=runner();r.model._exact_tail_installed=r.model._pfactor_agg_installed=True
        records,rows=self.report(r)
        self.assertTrue(all(x.levelname=='INFO' for x in records))
        self.assertTrue(all(x['installed'] and x['missing_switches']==[] for x in rows))

    def test_enabled_but_not_reached_is_visible(self):
        with patch.dict(os.environ,{key:'1' for key in g.SWITCHES}):
            _,rows=self.report(runner())
        self.assertTrue(all(x['status']=='not-installed' for x in rows))
        self.assertTrue(all('not reached' in x['reason'] for x in rows))

    def test_shallow_does_not_suggest_factor_only_or_fullN(self):
        records,rows=self.report(runner(shallow=True))
        self.assertEqual(records[0].levelname,'WARNING')
        self.assertNotIn(g.FACTOR_ONLY+'=1',rows[0]['missing_switches'])
        self.assertEqual(rows[1]['status'],'not-applicable')

    def test_D_and_AGG_do_not_suggest_P_only_exact_tail(self):
        for role in ('decode','null'):
            records,rows=self.report(runner(role))
            self.assertTrue(all(x.levelname=='INFO' for x in records))
            self.assertTrue(all(x['missing_switches']==[] for x in rows))

    def test_AGG_fullN_optin_without_supported_installer_warns(self):
        with patch.dict(os.environ,{g.AGG_FULL_N:'1'}):records,rows=self.report(runner('null'))
        self.assertEqual(records[0].levelname,'INFO')
        self.assertEqual(records[1].levelname,'WARNING')
        self.assertEqual(rows[1]['missing_switches'],[g.COMMIT+'=1'])

    def test_explicit_fullN_disable_preserves_switch_and_warns(self):
        r=runner();r.model._exact_tail_installed=True
        with patch.dict(os.environ,{g.EXACT:'1',g.COMMIT:'1',g.FACTOR_ONLY:'1',g.FULL_N:'0'}):
            records,rows=self.report(r)
        self.assertEqual(rows[1]['missing_switches'],[g.FULL_N+'=1'])
        self.assertEqual(os.environ,{})

    def test_nonfactor_default_has_no_receipt(self):
        with self.assertNoLogs(g.logger,level='INFO'):g.report_startup(runner(rank=0))

    def test_rejection_warns_once_preserves_exception_and_missing_flags(self):
        for route in ('exact-tail','full-N'):
            error=ValueError('native DUET rejected\nold legacy guard')
            original=Mock(side_effect=error);observed=g.warn_install_rejection(route)(original)
            with self.assertLogs(g.logger,level='WARNING') as cap:
                with self.assertRaises(ValueError) as got:observed(runner())
            self.assertIs(got.exception,error);original.assert_called_once()
            self.assertEqual(len(cap.records),1)
            row=messages(cap.records)[0];self.assertEqual(row['status'],'rejected')
            self.assertIn('old legacy guard',row['reason'])
            self.assertEqual(len(cap.records[0].getMessage().splitlines()),1)

    def test_successful_installer_return_and_mutations_unchanged(self):
        value=object();r=runner()
        def original(r,*args,**kw):r.model._exact_tail_installed=True;return value
        wrapped=g.warn_install_rejection('exact-tail')(original)
        with self.assertNoLogs(g.logger,level='WARNING'):self.assertIs(wrapped(r),value)
        self.assertTrue(r.model._exact_tail_installed)
        self.assertIs(wrapped.__wrapped__,original)

    def test_legacy_constructor_failure_and_disabled_no_log(self):
        runtime=ModuleType('sglang.srt.runtime_context');runtime.get_disagg=lambda:NS(disaggregation_mode='prefill')
        owner=runner().model;error=ValueError('factor-only must keep all 48 native layers without emitters')
        def original(owner):raise error
        wrapped=g.warn_factor_only_constructor(original)
        with patch.dict(sys.modules,{runtime.__name__:runtime}),patch.dict(os.environ,{g.FACTOR_ONLY:'1'}):
            with self.assertLogs(g.logger,level='WARNING') as cap:
                with self.assertRaises(ValueError) as got:wrapped(owner)
            self.assertIs(got.exception,error);self.assertEqual(messages(cap.records)[0]['status'],'constructor-rejected')
        with self.assertNoLogs(g.logger,level='WARNING'):
            with self.assertRaises(ValueError) as got:wrapped(owner)
        self.assertIs(got.exception,error)

    def test_constructor_success_does_not_touch_output_or_state(self):
        def original(owner,value):owner.value=value;return None
        wrapped=g.warn_factor_only_constructor(original);owner=NS();value=object()
        with patch.dict(os.environ,{g.FACTOR_ONLY:'1'}),self.assertNoLogs(g.logger,level='WARNING'):
            self.assertIsNone(wrapped(owner,value))
        self.assertIs(owner.value,value)

    def test_actual_model_runner_order_and_two_states(self):
        path=ROOT/'python/sglang/srt/model_executor/model_runner.py'
        for enabled in (False,True):
            events=[];r=runner();r.req_to_token_pool.factored_gdn_pool.prewarm_commit_graph=lambda:events.append('commit')
            r.req_to_token_pool.factored_gdn_pool.prewarm_k31_batch_graph=lambda:events.append('k31')
            exact=ModuleType('sglang.srt.mem_cache.gdn_prefill_exact_tail')
            fulln=ModuleType('sglang.srt.mem_cache.gdn_prefill_agg_contract')
            def install_exact(r):events.append('exact');r.model._exact_tail_installed=True
            def install_fulln(r):events.append('full-N');r.model._pfactor_agg_installed=True
            exact.install=install_exact;fulln.install=install_fulln;fulln.prewarm=lambda p:events.append('prewarm')
            capture=NS(eager_runner=object(),prefill=NS(runner=object()),decode=NS(runner=object()),memory_usage=13,time_usage=17)
            def capture_fn(**kwargs):events.append('capture');return capture
            scope=dict(os=os,capture_cuda_graphs=capture_fn)
            call=method(path,'ModelRunner','init_cuda_graphs',scope)
            with patch.dict(sys.modules,{exact.__name__:exact,fulln.__name__:fulln}),patch.dict(os.environ,{g.EXACT:str(int(enabled))}):
                with self.assertLogs(g.logger,level='INFO') as cap:call(r)
            self.assertEqual(events,['exact','full-N','commit','prewarm','k31','capture'] if enabled else ['commit','prewarm','k31','capture'])
            self.assertTrue(all(row['installed']==enabled for row in messages(cap.records)))
            self.assertIs(r.eager_runner,capture.eager_runner);self.assertEqual(r.graph_memory_usage,13)

    def test_actual_adapter_installs_constructor_observer_after_legacy(self):
        path=ROOT/'python/sglang/srt/models/flash_next_duet/pd_shallow_install.py'
        tree=ast.parse(path.read_text());install=next(n for n in tree.body if getattr(n,'name','')=='install')
        # Avoid importing a GPU model: only the actual installer body, with its pd module stub.
        name='sglang.srt.models.flash_next_duet';package=ModuleType(name);package.pd_shallow=NS()
        scope={'__package__':name,'wraps':__import__('functools').wraps,'os':os}
        contract=next(n for n in tree.body if getattr(n,'name','')=='factor_only_contract')
        exec(compile(ast.Module(body=[contract,install],type_ignores=[]),str(path),'exec'),scope)
        error=ValueError('factor-only must keep all 48 native layers without emitters')
        class Owner:
            prepare_before_cuda_graph_capture=lambda *a:None
            _twinstar_prefill=lambda *a:None
            _is_twinstar_prefill=lambda *a:True
            def __init__(self):self.fullstack={'gdn_rank':8}
        def legacy(cls,stock,**kwargs):
            self.assertIs(kwargs['factor_only_contract'],scope['factor_only_contract'])
            init=cls.__init__
            def rejected(self):init(self);raise error
            cls.__init__=rejected
        runtime=ModuleType('sglang.srt.runtime_context');runtime.get_disagg=lambda:NS(disaggregation_mode='prefill')
        with patch.dict(sys.modules,{name:package,runtime.__name__:runtime}),patch.dict(os.environ,{g.FACTOR_ONLY:'1'}):
            scope['install'](Owner,object(),NS(install=legacy))
            with self.assertLogs(g.logger,level='WARNING') as cap:
                with self.assertRaises(ValueError) as got:Owner()
        self.assertIs(got.exception,error);self.assertEqual(len(cap.records),1)

    def test_installer_bodies_and_graph_order_unchanged_except_logs(self):
        # Branch/arithmetic identity is checked against the base in the CPU driver.
        for name in ('gdn_prefill_exact_tail.py','gdn_prefill_agg_contract.py'):
            tree=ast.parse((ROOT/'python/sglang/srt/mem_cache'/name).read_text())
            install=next(n for n in tree.body if getattr(n,'name','')=='install')
            self.assertEqual(len(install.decorator_list),1)
            self.assertEqual(install.decorator_list[0].func.id,'warn_install_rejection')


if __name__=='__main__':unittest.main()
