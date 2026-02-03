#!/bin/bash
# 수정된 action space로 전체 파이프라인 재실행

echo "========================================================================"
echo "STEP 1: 데이터 재생성 (수정된 accumulation 계산 적용)"
echo "========================================================================"
echo ""
echo "현재 설정:"
echo "  - Propofol max: 12 mg/kg/h (이전: 30)"
echo "  - Remifentanil max: 2.0 μg/kg/min (이전: 1.0)"
echo "  - action_max: 12.0 (이전: 30.0)"
echo "  - Accumulation: rate × timestep_duration (이전: rate만)"
echo ""

read -p "데이터를 재생성하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "데이터 재생성 중..."
    python prepare_vitaldb_quick.py
    
    echo ""
    echo "========================================================================"
    echo "STEP 2: 재생성된 데이터 검증"
    echo "========================================================================"
    python check_data_ranges.py
    
    echo ""
    echo "========================================================================"
    echo "STEP 3: 모델 재학습 (새로운 action space로)"
    echo "========================================================================"
    echo ""
    echo "재학습 옵션:"
    echo "  1. 빠른 비교 (classical vs quantum, 간단)"
    echo "  2. 전체 비교 (experiments/compare_quantum_vs_classical_dualdrug.py)"
    echo "  3. Quantum만 재학습 (experiments/train_quantum.py)"
    echo ""
    read -p "선택하세요 (1/2/3): " -n 1 -r
    echo
    
    if [[ $REPLY == "1" ]]
    then
        echo "빠른 비교 실행 중..."
        python experiments/compare_quantum_vs_classical.py
    elif [[ $REPLY == "2" ]]
    then
        echo "전체 비교 실행 중..."
        python experiments/compare_quantum_vs_classical_dualdrug.py
    elif [[ $REPLY == "3" ]]
    then
        echo "Quantum 모델 학습 중..."
        python experiments/train_quantum.py
    else
        echo "잘못된 선택입니다."
    fi
else
    echo "작업이 취소되었습니다."
fi

echo ""
echo "========================================================================"
echo "완료!"
echo "========================================================================"
