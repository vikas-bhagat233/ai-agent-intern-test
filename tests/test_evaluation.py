import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_agent import RAGAgent
import openai
import os

class TestEvaluation:
    
    @pytest.fixture
    def agent(self):
        """Create a test agent"""
        openai.api_key = os.getenv("OPENAI_API_KEY", "test_key")
        return RAGAgent(
            knowledge_base_path="./knowledge-base",
            orders_path="./data/orders.json"
        )
    
    def load_test_cases(self, filename):
        """Load test cases from JSON"""
        path = Path(__file__).parent.parent / "evaluation" / filename
        with open(path, 'r') as f:
            return json.load(f)
    
    def test_visible_cases(self, agent):
        """Test all visible cases from the evaluation file"""
        cases_data = self.load_test_cases("visible-cases.json")
        cases = cases_data.get('cases', [])
        
        results = []
        for case in cases:
            agent.start_conversation()
            result = None
            for msg in case['messages']:
                if msg['role'] == 'user':
                    result = agent.process_message(msg['content'])
            
            # Check basic requirements
            test_result = {
                'case_id': case.get('id', 'unknown'),
                'passed': True,
                'checks': [],
                'response': result['response'][:200] + '...' if result else ''
            }
            
            if not result:
                test_result['passed'] = False
                results.append(test_result)
                continue
                
            expect = case.get('expect', {})
            response_lower = result['response'].lower()
            
            # Check must_include
            if 'must_include' in expect:
                for term in expect['must_include']:
                    term_present = term.lower() in response_lower
                    test_result['checks'].append({
                        'name': f"must_include_{term}",
                        'passed': term_present,
                        'message': f"Expected term '{term}' not found in response" if not term_present else "OK"
                    })
                    if not term_present:
                        test_result['passed'] = False
                        
            # Check must_not_include
            if 'must_not_include' in expect:
                for term in expect['must_not_include']:
                    term_absent = term.lower() not in response_lower
                    test_result['checks'].append({
                        'name': f"must_not_include_{term}",
                        'passed': term_absent,
                        'message': f"Forbidden term '{term}' found in response" if not term_absent else "OK"
                    })
                    if not term_absent:
                        test_result['passed'] = False
                        
            # Check required_sources
            if 'required_sources' in expect:
                source_filenames = [s['filename'] for s in result.get('sources', [])]
                for src in expect['required_sources']:
                    src_present = src in source_filenames
                    test_result['checks'].append({
                        'name': f"require_source_{src}",
                        'passed': src_present,
                        'message': f"Required source '{src}' not cited" if not src_present else "OK"
                    })
                    if not src_present:
                        test_result['passed'] = False
                        
            # Check forbidden_sources_as_authority
            if 'forbidden_sources_as_authority' in expect:
                source_filenames = [s['filename'] for s in result.get('sources', [])]
                for src in expect['forbidden_sources_as_authority']:
                    src_absent = src not in source_filenames
                    test_result['checks'].append({
                        'name': f"forbidden_source_{src}",
                        'passed': src_absent,
                        'message': f"Forbidden source '{src}' was cited" if not src_absent else "OK"
                    })
                    if not src_absent:
                        test_result['passed'] = False
                        
            # Check tool use
            if 'tool' in expect:
                expected_tool = expect['tool']
                tool_performed = result.get('order_lookup_performed', False)
                if expected_tool == 'order_lookup':
                    tool_ok = tool_performed
                    msg = "Expected order lookup tool to be called"
                elif expected_tool == 'not_called':
                    tool_ok = not tool_performed
                    msg = "Expected order lookup tool NOT to be called"
                else:
                    tool_ok = True  # optional or other values
                    msg = "OK"
                test_result['checks'].append({
                    'name': f"tool_{expected_tool}",
                    'passed': tool_ok,
                    'message': msg if not tool_ok else "OK"
                })
                if not tool_ok:
                    test_result['passed'] = False
                    
            # Check handoff
            if 'handoff' in expect:
                expected_handoff = expect['handoff']
                actual_handoff = result.get('needs_human_handoff', False)
                handoff_ok = (expected_handoff == actual_handoff)
                test_result['checks'].append({
                    'name': "handoff",
                    'passed': handoff_ok,
                    'message': f"Handoff expected to be {expected_handoff} but was {actual_handoff}" if not handoff_ok else "OK"
                })
                if not handoff_ok:
                    test_result['passed'] = False
            
            results.append(test_result)
            
        # Report results
        self._report_results(results, "Visible Cases")
        
        # Assert all passed
        assert all(r['passed'] for r in results), "Some visible cases failed"
    
    def test_custom_cases(self, agent):
        """Test custom evaluation cases"""
        cases = self.load_test_cases("custom-cases.json")
        agent.start_conversation()
        
        results = []
        for case in cases:
            # Process each turn in multi-turn cases
            if case.get('multi_turn'):
                agent.start_conversation()
                turn_results = []
                for turn in case['turns']:
                    result = agent.process_message(turn['prompt'])
                    turn_results.append({
                        'turn': turn.get('id', ''),
                        'response': result['response'][:200] + '...',
                        'sources': result.get('sources', []),
                        'needs_human': result.get('needs_human_handoff', False)
                    })
                results.append({
                    'case_id': case.get('id', 'unknown'),
                    'type': 'multi_turn',
                    'turns': turn_results,
                    'passed': True
                })
            else:
                # Single turn case
                result = agent.process_message(case['prompt'])
                
                test_result = {
                    'case_id': case.get('id', 'unknown'),
                    'passed': True,
                    'checks': [],
                    'response': result['response'][:200] + '...'
                }
                
                # Check based on expected behavior
                if case.get('expected_behavior'):
                    expected = case['expected_behavior']
                    
                    if 'should_ask_for_order_id' in expected:
                        should_ask = 'order id' in result['response'].lower()
                        test_result['checks'].append({
                            'name': 'ask_for_order_id',
                            'passed': should_ask,
                            'message': 'Should ask for order ID' if should_ask else 'OK'
                        })
                        if not should_ask:
                            test_result['passed'] = False
                    
                    if 'should_use_order_lookup' in expected:
                        used_lookup = result.get('order_lookup_performed', False)
                        test_result['checks'].append({
                            'name': 'use_order_lookup',
                            'passed': used_lookup,
                            'message': 'Should use order lookup' if used_lookup else 'OK'
                        })
                        if not used_lookup:
                            test_result['passed'] = False
                    
                    if 'should_cite_sources' in expected:
                        has_sources = len(result.get('sources', [])) > 0
                        test_result['checks'].append({
                            'name': 'cite_sources',
                            'passed': has_sources,
                            'message': 'Should cite sources' if has_sources else 'OK'
                        })
                        if not has_sources:
                            test_result['passed'] = False
                
                results.append(test_result)
        
        # Report results
        self._report_results(results, "Custom Cases")
        
        # Assert all passed
        assert all(r['passed'] for r in results), "Some custom cases failed"
    
    def _report_results(self, results, category):
        """Report evaluation results"""
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {category}")
        print(f"{'='*60}")
        
        total = len(results)
        passed = sum(1 for r in results if r['passed'])
        
        print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")
        print(f"Pass Rate: {passed/total*100:.1f}%\n")
        
        # Show details for failed cases
        for result in results:
            if not result['passed']:
                print(f"❌ Case {result.get('case_id', 'unknown')} failed")
                for check in result.get('checks', []):
                    if not check['passed']:
                        print(f"   - {check['name']}: {check['message']}")

def test_evaluation_suite():
    """Run the full evaluation suite"""
    # This is the main entry point for evaluation
    import pytest
    pytest.main([__file__, '-v'])