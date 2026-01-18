"""Helper script to run sections of step3b_workflow.py individually"""
import sys
import re
from pathlib import Path

def extract_sections(file_path):
    """Extract sections marked with # %% from the workflow file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by # %% markers
    sections = []
    current_section = []
    current_markdown = []
    
    for line in content.split('\n'):
        if line.strip().startswith('# %%'):
            if current_section or current_markdown:
                sections.append({
                    'type': 'markdown' if current_markdown else 'code',
                    'content': '\n'.join(current_markdown if current_markdown else current_section)
                })
            current_section = []
            current_markdown = []
            if '[markdown]' in line:
                current_markdown.append(line)
            else:
                current_section.append(line)
        else:
            if current_markdown:
                current_markdown.append(line)
            elif current_section or line.strip():
                current_section.append(line)
    
    # Add last section
    if current_section or current_markdown:
        sections.append({
            'type': 'markdown' if current_markdown else 'code',
            'content': '\n'.join(current_markdown if current_markdown else current_section)
        })
    
    return sections

def run_section(section_num, file_path='step3b_workflow.py'):
    """Run a specific section (1-indexed)"""
    sections = extract_sections(file_path)
    
    if section_num < 1 or section_num > len(sections):
        print(f"Section {section_num} not found. Available sections: 1-{len(sections)}")
        return False
    
    section = sections[section_num - 1]
    
    if section['type'] == 'markdown':
        print(f"Section {section_num} is a markdown cell (skipping execution)")
        print("=" * 80)
        print(section['content'][:200] + "..." if len(section['content']) > 200 else section['content'])
        print("=" * 80)
        return True
    
    print(f"\n{'='*80}")
    print(f"Running Section {section_num}")
    print(f"{'='*80}\n")
    
    # Create a namespace for execution (accumulates variables across sections)
    if not hasattr(run_section, 'namespace'):
        run_section.namespace = {'__name__': '__main__', '__file__': file_path}
        # Set up UTF-8 encoding in the namespace
        if sys.platform == 'win32':
            import io
            run_section.namespace['sys'] = sys
            # Don't wrap stdout here, let the section code do it if needed
    
    # Execute the section in the accumulated namespace
    try:
        exec(section['content'], run_section.namespace)
        print(f"\n✅ Section {section_num} completed successfully")
        return True
    except Exception as e:
        print(f"\n❌ Section {section_num} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sections = extract_sections('step3b_workflow.py')
        print(f"Available sections: 1-{len(sections)}")
        print("\nUsage: python run_section.py <section_number>")
        print("Example: python run_section.py 1")
    else:
        section_num = int(sys.argv[1])
        run_section(section_num)
