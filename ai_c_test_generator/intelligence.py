"""
Test Intelligence Analyzer - Provides advanced analysis of test failures and fixes
"""

import os
import re
from typing import Dict, List, Tuple
from pathlib import Path
import json

import google.generativeai as genai


class TestIntelligenceAnalyzer:
    """AI-powered test intelligence analyzer for root cause identification and fix guidance"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def analyze_test_failures(self, test_results: Dict, source_files: List[str]) -> Dict:
        """
        Analyze test execution results and provide intelligence report

        Args:
            test_results: Dictionary containing test execution results
            source_files: List of source files being tested

        Returns:
            Intelligence report with root causes, fixes, and prioritization
        """
        intelligence_report = {
            'executive_summary': {},
            'priority_fixes': [],
            'detailed_analysis': [],
            'quality_metrics': {},
            'roi_analysis': {},
            'recommendations': []
        }

        # Analyze each test result
        for test_name, result in test_results.items():
            if not result.get('passed', True):
                analysis = self._analyze_single_failure(test_name, result, source_files)
                intelligence_report['detailed_analysis'].append(analysis)

        # Generate priority fixes
        intelligence_report['priority_fixes'] = self._prioritize_fixes(intelligence_report['detailed_analysis'])

        # Calculate quality metrics
        intelligence_report['quality_metrics'] = self._calculate_quality_metrics(test_results, intelligence_report['detailed_analysis'])

        # Generate executive summary
        intelligence_report['executive_summary'] = self._generate_executive_summary(intelligence_report)

        # Calculate ROI
        intelligence_report['roi_analysis'] = self._calculate_roi(intelligence_report)

        return intelligence_report

    def _analyze_single_failure(self, test_name: str, result: Dict, source_files: List[str]) -> Dict:
        """Analyze a single test failure and provide detailed intelligence"""

        # Extract error information
        error_output = result.get('error_output', '')
        compilation_errors = result.get('compilation_errors', [])

        # Use AI to analyze the failure
        analysis_prompt = f"""
        Analyze this C unit test failure and provide detailed intelligence:

        Test Name: {test_name}
        Error Output: {error_output}
        Compilation Errors: {json.dumps(compilation_errors, indent=2)}

        Source Files: {', '.join(source_files)}

        Provide analysis in this exact JSON format:
        {{
            "root_cause": "Brief description of why the test failed",
            "error_category": "COMPILATION|LOGIC|RUNTIME|DEPENDENCY",
            "severity": "CRITICAL|HIGH|MEDIUM|LOW",
            "fix_complexity": "EASY|MEDIUM|HARD",
            "estimated_fix_time": "X minutes",
            "fix_instructions": ["Step 1", "Step 2", "Step 3"],
            "impact_assessment": "What this fix achieves",
            "code_changes_required": "Brief description of changes needed",
            "prerequisites": ["Any prerequisites for the fix"],
            "alternative_solutions": ["Alternative approaches if primary fix fails"]
        }}
        """

        try:
            response = self.model.generate_content(analysis_prompt)
            analysis = json.loads(response.text.strip('```json\n').strip('```'))
        except Exception as e:
            # Fallback analysis if AI fails
            analysis = self._fallback_analysis(test_name, result)

        analysis['test_name'] = test_name
        analysis['confidence_score'] = self._calculate_confidence_score(analysis)

        return analysis

    def _fallback_analysis(self, test_name: str, result: Dict) -> Dict:
        """Provide basic analysis when AI analysis fails"""
        error_output = result.get('error_output', '').lower()

        analysis = {
            "root_cause": "Unable to determine automatically - manual review required",
            "error_category": "UNKNOWN",
            "severity": "MEDIUM",
            "fix_complexity": "MEDIUM",
            "estimated_fix_time": "15 minutes",
            "fix_instructions": ["Review error output manually", "Check compilation errors", "Verify test logic"],
            "impact_assessment": "Manual debugging required",
            "code_changes_required": "TBD - requires manual analysis",
            "prerequisites": [],
            "alternative_solutions": ["Consult C programming documentation", "Review similar test patterns"]
        }

        # Basic pattern matching for common errors
        if 'undefined reference' in error_output:
            analysis.update({
                "root_cause": "Missing function definition or incorrect linking",
                "error_category": "DEPENDENCY",
                "fix_complexity": "MEDIUM"
            })
        elif 'expected' in error_output and 'but was' in error_output:
            analysis.update({
                "root_cause": "Test assertion failure - actual vs expected values don't match",
                "error_category": "LOGIC",
                "fix_complexity": "EASY"
            })
        elif 'void value not ignored' in error_output:
            analysis.update({
                "root_cause": "Attempting to assign void function result",
                "error_category": "COMPILATION",
                "fix_complexity": "EASY"
            })

        return analysis

    def _prioritize_fixes(self, detailed_analysis: List[Dict]) -> List[Dict]:
        """Prioritize fixes based on complexity, impact, and severity"""

        # Calculate priority score for each fix
        for analysis in detailed_analysis:
            priority_score = self._calculate_priority_score(analysis)
            analysis['priority_score'] = priority_score

        # Sort by priority score (higher is better)
        sorted_analysis = sorted(detailed_analysis, key=lambda x: x['priority_score'], reverse=True)

        # Group by complexity
        priority_fixes = []
        for analysis in sorted_analysis:
            priority_fixes.append({
                'test_name': analysis['test_name'],
                'complexity': analysis['fix_complexity'],
                'estimated_time': analysis['estimated_fix_time'],
                'impact': analysis['impact_assessment'],
                'root_cause': analysis['root_cause'],
                'fix_instructions': analysis['fix_instructions'][:3],  # Top 3 steps
                'priority_score': analysis['priority_score']
            })

        return priority_fixes

    def _calculate_priority_score(self, analysis: Dict) -> float:
        """Calculate priority score based on multiple factors"""

        # Base scores
        severity_scores = {'CRITICAL': 100, 'HIGH': 75, 'MEDIUM': 50, 'LOW': 25}
        complexity_scores = {'EASY': 80, 'MEDIUM': 50, 'HARD': 20}

        severity_score = severity_scores.get(analysis.get('severity', 'MEDIUM'), 50)
        complexity_score = complexity_scores.get(analysis.get('complexity', 'MEDIUM'), 50)

        # Extract time estimate
        time_match = re.search(r'(\d+)', analysis.get('estimated_fix_time', '15'))
        time_estimate = int(time_match.group(1)) if time_match else 15

        # Time bonus (faster fixes get higher priority)
        time_bonus = max(0, 20 - time_estimate)  # Max 20 points for very quick fixes

        # Confidence bonus
        confidence_bonus = analysis.get('confidence_score', 50) * 0.2

        total_score = severity_score + complexity_score + time_bonus + confidence_bonus

        return round(total_score, 1)

    def _calculate_quality_metrics(self, test_results: Dict, detailed_analysis: List[Dict]) -> Dict:
        """Calculate overall quality metrics for the test suite"""

        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result.get('passed', False))
        failed_tests = total_tests - passed_tests

        # Calculate quality score
        base_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        # Penalty for complexity of failures
        complexity_penalty = 0
        for analysis in detailed_analysis:
            if analysis['fix_complexity'] == 'HARD':
                complexity_penalty += 5
            elif analysis['fix_complexity'] == 'MEDIUM':
                complexity_penalty += 2

        quality_score = max(0, base_score - complexity_penalty)

        # Estimate coverage potential
        coverage_potential = min(95, quality_score + 10)  # Assume 10% improvement potential

        # Calculate maintenance complexity
        maintenance_complexity = 'LOW'
        if complexity_penalty > 20:
            maintenance_complexity = 'HIGH'
        elif complexity_penalty > 10:
            maintenance_complexity = 'MEDIUM'

        return {
            'quality_score': round(quality_score, 1),
            'coverage_potential': round(coverage_potential, 1),
            'maintenance_complexity': maintenance_complexity,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'failure_rate': round((failed_tests / total_tests) * 100, 1) if total_tests > 0 else 0
        }

    def _generate_executive_summary(self, intelligence_report: Dict) -> Dict:
        """Generate executive summary with key insights"""

        quality_metrics = intelligence_report['quality_metrics']
        priority_fixes = intelligence_report['priority_fixes']

        # Calculate time savings
        total_fix_time = sum(int(re.search(r'(\d+)', fix['estimated_time']).group(1))
                           for fix in priority_fixes[:5])  # Top 5 fixes

        # Estimate manual debugging time (assume 30 min per failure without intelligence)
        manual_debugging_time = quality_metrics['failed_tests'] * 30

        time_savings = manual_debugging_time - total_fix_time

        return {
            'total_time_savings': f"{time_savings} minutes",
            'recommended_fix_order': [fix['complexity'] for fix in priority_fixes[:3]],
            'expected_final_coverage': quality_metrics['coverage_potential'],
            'quality_improvement_potential': f"{quality_metrics['coverage_potential'] - quality_metrics['quality_score']:.1f} points",
            'top_priority_fixes': len([f for f in priority_fixes if f['complexity'] == 'EASY'])
        }

    def _calculate_roi(self, intelligence_report: Dict) -> Dict:
        """Calculate return on investment metrics"""

        exec_summary = intelligence_report['executive_summary']
        quality_metrics = intelligence_report['quality_metrics']

        # Assume engineering cost of $50/hour
        hourly_rate = 50
        time_savings_hours = int(re.search(r'(\d+)', exec_summary['total_time_savings']).group(1)) / 60
        dollar_savings = time_savings_hours * hourly_rate

        # Calculate quality improvement ROI
        quality_improvement = float(exec_summary['quality_improvement_potential'].split()[0])
        quality_roi = quality_improvement * 10  # Assume $10 value per quality point

        return {
            'engineering_time_saved': f"${dollar_savings:.0f}",
            'quality_improvement_value': f"${quality_roi:.0f}",
            'total_roi': f"${dollar_savings + quality_roi:.0f}",
            'debugging_efficiency': f"{quality_metrics['failure_rate']:.1f}% faster debugging"
        }

    def _calculate_confidence_score(self, analysis: Dict) -> int:
        """Calculate confidence score for the analysis (0-100)"""

        confidence = 50  # Base confidence

        # Boost confidence based on analysis quality
        if analysis.get('root_cause') and 'Unable to determine' not in analysis['root_cause']:
            confidence += 20

        if analysis.get('fix_instructions') and len(analysis['fix_instructions']) > 0:
            confidence += 15

        if analysis.get('error_category') != 'UNKNOWN':
            confidence += 10

        if analysis.get('estimated_fix_time') and 'TBD' not in analysis['estimated_fix_time']:
            confidence += 5

        return min(100, confidence)

    def generate_intelligence_report(self, output_path: str, intelligence_report: Dict):
        """Generate a comprehensive markdown intelligence report"""

        report_content = f"""# Test Generation Intelligence Report

## Executive Summary
- **Total time savings vs. manual**: {intelligence_report['executive_summary']['total_time_savings']}
- **Recommended fix order**: {' → '.join(intelligence_report['executive_summary']['recommended_fix_order'])}
- **Expected final coverage**: {intelligence_report['executive_summary']['expected_final_coverage']}%

## Quality Metrics
- **Code Quality Score**: {intelligence_report['quality_metrics']['quality_score']}/100
- **Test Coverage Potential**: {intelligence_report['quality_metrics']['coverage_potential']}%
- **Maintenance Complexity**: {intelligence_report['quality_metrics']['maintenance_complexity']}
- **Failure Rate**: {intelligence_report['quality_metrics']['failure_rate']}%

## ROI Analysis
- **Engineering time saved**: {intelligence_report['roi_analysis']['engineering_time_saved']}
- **Quality improvement value**: {intelligence_report['roi_analysis']['quality_improvement_value']}
- **Total ROI**: {intelligence_report['roi_analysis']['total_roi']}
- **Debugging efficiency**: {intelligence_report['roi_analysis']['debugging_efficiency']}

## Priority Fixes (Under 10 minutes total)

"""

        # Add priority fixes
        easy_fixes = [f for f in intelligence_report['priority_fixes'] if f['complexity'] == 'EASY']
        for i, fix in enumerate(easy_fixes[:5], 1):
            report_content += f"""### {i}. {fix['test_name']} - {fix['estimated_time']}
**Root Cause**: {fix['root_cause']}
**Impact**: {fix['impact']}
**Fix Steps**:
"""
            for j, step in enumerate(fix['fix_instructions'], 1):
                report_content += f"  {j}. {step}\n"
            report_content += "\n"

        # Add detailed analysis
        report_content += "## Detailed Analysis\n\n"
        for analysis in intelligence_report['detailed_analysis']:
            report_content += f"""### {analysis['test_name']}
- **Root Cause**: {analysis['root_cause']}
- **Category**: {analysis['error_category']}
- **Severity**: {analysis['severity']}
- **Complexity**: {analysis['fix_complexity']}
- **Estimated Time**: {analysis['estimated_fix_time']}
- **Confidence**: {analysis['confidence_score']}%

**Fix Instructions**:
"""
            for step in analysis['fix_instructions']:
                report_content += f"- {step}\n"

            report_content += f"""
**Impact**: {analysis['impact_assessment']}
**Code Changes**: {analysis['code_changes_required']}

**Prerequisites**:
"""
            for prereq in analysis.get('prerequisites', []):
                report_content += f"- {prereq}\n"

            report_content += "\n**Alternative Solutions**:\n"
            for alt in analysis.get('alternative_solutions', []):
                report_content += f"- {alt}\n"
            report_content += "\n---\n\n"

        # Write report
        with open(output_path, 'w') as f:
            f.write(report_content)

    def generate_fix_priority_csv(self, output_path: str, intelligence_report: Dict):
        """Generate CSV file with fix priorities"""

        import csv

        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['test_name', 'complexity', 'estimated_time', 'priority_score', 'root_cause', 'impact']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for fix in intelligence_report['priority_fixes']:
                writer.writerow({
                    'test_name': fix['test_name'],
                    'complexity': fix['complexity'],
                    'estimated_time': fix['estimated_time'],
                    'priority_score': fix['priority_score'],
                    'root_cause': fix['root_cause'],
                    'impact': fix['impact']
                })</content>
<parameter name="filePath">c:\Users\SwathantraPulicherla\OneDrive - requisimus Holding GmbH\Desktop\AI_tools_project\workspaces\ai-tool-lab\ai-tool-gen-lab\ai_c_test_generator\intelligence.py