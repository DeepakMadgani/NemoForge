import pytest
from ticket_to_code.agents.smart_extract import (
    _parse_typescript_boundaries_treesitter,
    smart_extract,
    extract_exact_methods,
    MethodBoundary,
)

SAMPLE_TS_COMPONENT = """import { Component, OnInit } from '@angular/core';
import { Subject } from 'rxjs';

@Component({
  selector: 'app-sample',
  template: '<div>Sample</div>'
})
export class SampleComponent implements OnInit {
  // Class fields before constructor
  title: string = 'Hello World';
  isLoaded: boolean = false;
  items: string[] = ['a', 'b'];
  private count = 42;
  config = { enabled: true };
  onDestroy$ = new Subject<void>();

  constructor() {
    console.log('Constructor called');
  }

  ngOnInit(): void {
    this.isLoaded = true;
  }

  loadData(): void {
    this.title = 'Updated';
  }
}
"""

SAMPLE_TS_NO_CALLABLE = """export class DataModel {
  id: string;
  name: string;
  value: number;
}
"""

SAMPLE_TS_ARROW_FUNCTION = """export class HandlerComponent {
  fieldA: string = 'test';
  handleClick = () => {
    console.log('Clicked');
  };
  fieldB: number = 10;
}
"""


def test_ts_class_declaration_not_treated_as_first_callable():
    """Verify class declaration (kind='class') and fields (kind='property') are not callable."""
    boundaries = _parse_typescript_boundaries_treesitter(SAMPLE_TS_COMPONENT)
    assert len(boundaries) > 0
    assert boundaries[0].kind == "class"
    assert boundaries[0].name == "SampleComponent"

    # Properties should have kind="property"
    prop_boundaries = [b for b in boundaries if b.name in ("title", "isLoaded", "items", "count", "config", "onDestroy$")]
    assert len(prop_boundaries) >= 6
    for pb in prop_boundaries:
        assert pb.kind == "property", f"Expected property kind for {pb.name}, got {pb.kind}"

    # Constructor should be first callable
    callables = [b for b in boundaries if b.kind in ("method", "constructor", "function")]
    assert len(callables) >= 3
    first_callable = callables[0]
    assert first_callable.kind == "constructor"
    assert first_callable.name == "constructor"


def test_class_fields_included_in_exact_source():
    """Verify all class fields between class declaration and first method are included in exact source."""
    verbatim_content, matched = extract_exact_methods(
        content=SAMPLE_TS_COMPONENT,
        file_path="src/app/sample.component.ts",
        anchor_methods=["ngOnInit"],
    )

    # Class fields must be present verbatim in the header section
    assert "title: string = 'Hello World';" in verbatim_content
    assert "isLoaded: boolean = false;" in verbatim_content
    assert "items: string[] = ['a', 'b'];" in verbatim_content
    assert "private count = 42;" in verbatim_content
    assert "config = { enabled: true };" in verbatim_content
    assert "onDestroy$ = new Subject<void>();" in verbatim_content
    assert "export class SampleComponent implements OnInit {" in verbatim_content

    # Target method ngOnInit must be extracted
    assert any(m.name == "ngOnInit" for m in matched)
    assert "this.isLoaded = true;" in verbatim_content


def test_constructor_recognized_correctly():
    """Verify constructor boundary is identified with kind='constructor'."""
    boundaries = _parse_typescript_boundaries_treesitter(SAMPLE_TS_COMPONENT)
    ctor_b = next((b for b in boundaries if b.name == "constructor"), None)
    assert ctor_b is not None
    assert ctor_b.kind == "constructor"
    assert "constructor()" in ctor_b.signature


def test_arrow_function_recognized_as_callable():
    """Verify arrow functions defined as fields are recognized as callable methods."""
    boundaries = _parse_typescript_boundaries_treesitter(SAMPLE_TS_ARROW_FUNCTION)
    field_a = next(b for b in boundaries if b.name == "fieldA")
    assert field_a.kind == "property"

    handle_click = next(b for b in boundaries if b.name == "handleClick")
    assert handle_click.kind == "method"

    callables = [b for b in boundaries if b.kind in ("method", "constructor", "function")]
    assert len(callables) == 1
    assert callables[0].name == "handleClick"


def test_fallback_when_no_callable_exists():
    """Verify fallback behavior when a file has no methods/constructors."""
    verbatim_content, matched = extract_exact_methods(
        content=SAMPLE_TS_NO_CALLABLE,
        file_path="src/app/data.model.ts",
        anchor_methods=["nonExistentMethod"],
    )
    # Since no methods matched, matched is empty
    assert matched == []
    assert verbatim_content == ""

    # smart_extract should still provide the header up to fallback line
    outline = smart_extract(
        content=SAMPLE_TS_NO_CALLABLE,
        file_path="src/app/data.model.ts",
    )
    assert "export class DataModel {" in outline
    assert "id: string;" in outline


def test_existing_method_extraction_remains_unchanged():
    """Verify standard methods are extracted with exact body lines."""
    verbatim_content, matched = extract_exact_methods(
        content=SAMPLE_TS_COMPONENT,
        file_path="src/app/sample.component.ts",
        anchor_methods=["loadData"],
    )
    assert any(m.name == "loadData" for m in matched)
    assert "this.title = 'Updated';" in verbatim_content
