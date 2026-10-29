import tempfile
from pathlib import Path
from ticket_to_code.runtime.sibling_crawler import SiblingCrawler


def test_sibling_crawler_discovers_repositories_and_rosetta_joins():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        svc = root / "project-service"
        repo_dir = svc / "src" / "main" / "java" / "com" / "example" / "repository"
        model_dir = svc / "src" / "main" / "java" / "com" / "example" / "model"
        service_dir = svc / "src" / "main" / "java" / "com" / "example" / "service"
        repo_dir.mkdir(parents=True)
        model_dir.mkdir(parents=True)
        service_dir.mkdir(parents=True)

        # 1. Write Service with injected repo
        service_code = """package com.example.service;
import com.example.repository.ContractMemberRepository;
import lombok.RequiredArgsConstructor;
import java.util.UUID;

@RequiredArgsConstructor
public class ContractMemberDeleteService {
    private final ContractMemberRepository contractMemberRepository;
    public void deleteContractMember(UUID id) {
        contractMemberRepository.deleteById(id);
    }
}
"""
        service_file = service_dir / "ContractMemberDeleteService.java"
        service_file.write_text(service_code, encoding="utf-8")

        # 2. Write ContractMemberRepository with queries and methods
        repo_code = """package com.example.repository;
import com.example.model.ContractMember;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import java.util.UUID;

public interface ContractMemberRepository extends JpaRepository<ContractMember, UUID> {
    @Query("select count(cm) from ContractMember cm INNER JOIN ParticipatingMember pm ON pm.otdsUserId = cm.userId where cm.contract.id = :contractId")
    long countByContractId(@Param("contractId") UUID contractId);

    void deleteByContractId(UUID contractId);
}
"""
        repo_file = repo_dir / "ContractMemberRepository.java"
        repo_file.write_text(repo_code, encoding="utf-8")

        # 3. Write ContractMember model
        model_code = """package com.example.model;
import java.util.UUID;
import jakarta.persistence.*;

@Entity
public class ContractMember {
    @Id
    private UUID id;
    @Column(name = "user_id")
    private UUID userId;
}
"""
        model_file = model_dir / "ContractMember.java"
        model_file.write_text(model_code, encoding="utf-8")

        # 4. Crawl
        grounding = SiblingCrawler.crawl_service_siblings(root, str(service_file))
        assert "ContractMemberRepository.java" in grounding
        assert "JpaRepository<ContractMember, UUID>" in grounding
        assert "pm.otdsUserId <-> cm.userId" in grounding
        assert "countByContractId" in grounding
        assert "deleteByContractId" in grounding
        assert "ContractMember.java" in grounding
        assert "PrimaryKey=`UUID id`" in grounding
        assert "userId: UUID (`user_id`)" in grounding
