from ticket_to_code.intelligence.skeleton_compressor import SkeletonCompressor, SkeletonResult

def test_compress_typescript_angular():
    ts_code = """import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

@Component({
  selector: 'app-document-register',
  templateUrl: './document-register.component.html',
  styleUrls: ['./document-register.component.scss']
})
export class DocumentRegisterComponent implements OnInit {
  uploadForm: FormGroup;
  selectedFile?: File;

  constructor(private fb: FormBuilder) {
    this.uploadForm = this.fb.group({
      fileName: ['', Validators.required],
      fileType: ['PDF', Validators.required]
    });
  }

  ngOnInit(): void {
    console.log("Initialized component with 50 lines of complex logic");
    const a = 1;
    const b = 2;
    this.setupListeners();
  }

  submit(): void {
    if (this.uploadForm.invalid) {
      return;
    }
    this.uploadFile(this.selectedFile);
  }

  private mapUploadError(error: any): string {
    return error ? error.message : "Unknown error";
  }
}
"""
    result = SkeletonCompressor.compress_file("document-register.component.ts", content=ts_code)
    assert result.language == "typescript"
    assert result.compressed_chars < result.original_chars
    assert "uploadForm: FormGroup;" in result.skeleton_code
    assert "constructor(private fb: FormBuilder) { /* elided */ }" in result.skeleton_code
    assert "ngOnInit(): void { /* elided */ }" in result.skeleton_code
    assert "submit(): void { /* elided */ }" in result.skeleton_code
    assert "private mapUploadError(error: any): string { /* elided */ }" in result.skeleton_code
    assert "console.log" not in result.skeleton_code
    assert result.compression_ratio < 0.8


def test_compress_java_spring():
    java_code = """package com.cloudforge.document.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import javax.validation.Valid;

@RestController
@RequestMapping("/api/v1/documents")
public class RegisterDocumentController {

    private final DocumentService documentService;

    public RegisterDocumentController(DocumentService documentService) {
        this.documentService = documentService;
    }

    @PostMapping("/register")
    public ResponseEntity<DocumentResponse> register(@Valid @RequestBody RegisterDocumentRequest request) {
        // Complex 80-line transactional business logic
        log.info("Registering document: {}", request.getName());
        ValidationResult vr = validator.validate(request);
        if (!vr.isValid()) {
            throw new InvalidDocumentException(vr.getErrors());
        }
        Document doc = documentService.save(request.toEntity());
        return ResponseEntity.ok(new DocumentResponse(doc.getId(), "SUCCESS"));
    }
}
"""
    result = SkeletonCompressor.compress_file("RegisterDocumentController.java", content=java_code)
    assert result.language == "java"
    assert "@RestController" in result.skeleton_code
    assert "public class RegisterDocumentController" in result.skeleton_code
    assert "public RegisterDocumentController(DocumentService documentService) { /* elided */ }" in result.skeleton_code
    assert "public ResponseEntity<DocumentResponse> register(@Valid @RequestBody RegisterDocumentRequest request) { /* elided */ }" in result.skeleton_code
    assert "ValidationResult vr" not in result.skeleton_code
    assert result.compression_ratio < 0.75


def test_compress_html_angular():
    html_code = """<div class="deliverable-container">
  <table class="grid-table">
    <thead>
      <tr><th>Name</th><th>Issues</th><th>Action</th></tr>
    </thead>
    <tbody>
""" + "\n".join(f'      <tr><td>Item {i}</td><td>0</td><td><ot-icon-button></ot-icon-button></td></tr>' for i in range(25)) + """
    </tbody>
  </table>
</div>
"""
    result = SkeletonCompressor.compress_file("deliverable.component.html", content=html_code)
    assert result.language == "html"
    assert result.compressed_chars < result.original_chars
    assert "deliverable-container" in result.skeleton_code
    assert "<thead>" in result.skeleton_code
    assert "<tbody>" in result.skeleton_code
    assert "</tbody>" in result.skeleton_code
    assert "<!-- ... elided" in result.skeleton_code


def test_compress_yaml_gateway():
    yaml_code = """spring:
  cloud:
    gateway:
      routes:
        - id: issues-service
          uri: lb://issues-service
          predicates:
            - Path=/api/v1/issues/**
          filters:
            - StripPrefix=1
            - AddRequestHeader=X-Tenant, cloudforge
""" + "\n".join(f"            - DeepConfigKey{i}: value{i}" for i in range(30))
    result = SkeletonCompressor.compress_file("application.yml", content=yaml_code)
    assert result.language == "yaml"
    assert result.compressed_chars < result.original_chars
    assert "spring:" in result.skeleton_code
    assert "gateway:" in result.skeleton_code
    assert "- id: issues-service" in result.skeleton_code
    assert "- Path=/api/v1/issues/**" in result.skeleton_code
    assert "DeepConfigKey29" not in result.skeleton_code

