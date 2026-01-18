"""Run step3b_workflow.py sections sequentially"""
import sys
import re
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def run_workflow_sections(start_section=1, end_section=None):
    """Run workflow sections sequentially"""
    workflow_file = Path(__file__).parent / 'step3b_workflow.py'
    
    with open(workflow_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by # %% markers
    parts = re.split(r'^# %%', content, flags=re.MULTILINE)
    
    # First part is before any cells, skip it
    sections = parts[1:] if len(parts) > 1 else []
    
    if end_section is None:
        end_section = len(sections)
    
    print(f"Running sections {start_section} to {end_section} of {len(sections)}")
    print("=" * 80)
    
    # Create a namespace that persists across sections
    namespace = {
        '__name__': '__main__',
        '__file__': str(workflow_file),
    }
    
    for i in range(start_section - 1, min(end_section, len(sections))):
        section_num = i + 1
        section_content = sections[i]
        
        # Skip markdown-only sections
        if section_content.strip().startswith('[markdown]'):
            print(f"\n[Skipping Section {section_num} - Markdown]")
            continue
        
        print(f"\n{'='*80}")
        print(f"Running Section {section_num}")
        print(f"{'='*80}\n")
        
        try:
            exec(section_content, namespace)
            print(f"\n✅ Section {section_num} completed")
        except KeyboardInterrupt:
            print(f"\n⚠️  Section {section_num} interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Section {section_num} failed: {e}")
            import traceback
            traceback.print_exc()
            response = input("\nContinue to next section? (y/n): ")
            if response.lower() != 'y':
                break
    
    print(f"\n{'='*80}")
    print("Workflow execution completed")
    print(f"{'='*80}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        start = int(sys.argv[1])
        end = int(sys.argv[2]) if len(sys.argv) > 2 else None
        run_workflow_sections(start, end)
    else:
        print("Usage: python run_workflow_sections.py <start_section> [end_section]")
        print("Example: python run_workflow_sections.py 1 5")
        print("\nRunning all sections...")
        run_workflow_sections()
