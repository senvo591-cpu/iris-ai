/* =========================================================
   IRIS AI - FRONTEND
   ========================================================= */


function getNumber(id) {

    const element =
        document.getElementById(id);

    if (!element) {
        return null;
    }

    return parseFloat(
        element.value
    );
}


/* =========================================================
   VALIDATION
   ========================================================= */

function clearErrors() {

    const ids = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ];

    ids.forEach(id => {

        const error =
            document.getElementById(
                "error_" + id
            );

        const input =
            document.getElementById(id);

        if (error) {
            error.textContent = "";
        }

        if (input) {
            input.classList.remove(
                "input-error"
            );
        }

    });

}


function showError(
    id,
    message
) {

    const error =
        document.getElementById(
            "error_" + id
        );

    const input =
        document.getElementById(id);

    if (error) {
        error.textContent =
            message;
    }

    if (input) {
        input.classList.add(
            "input-error"
        );
    }

}


function validateInputs() {

    clearErrors();

    let valid = true;

    const ids = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ];

    ids.forEach(id => {

        const value =
            getNumber(id);

        if (
            value === null ||
            Number.isNaN(value)
        ) {

            showError(
                id,
                "Vui lòng nhập giá trị."
            );

            valid = false;

        }
        else if (value <= 0) {

            showError(
                id,
                "Giá trị phải lớn hơn 0."
            );

            valid = false;

        }

    });

    return valid;
}


/* =========================================================
   SAMPLE DATA
   ========================================================= */

function fillSample() {

    document.getElementById(
        "sepal_length"
    ).value = "5.1";


    document.getElementById(
        "sepal_width"
    ).value = "3.5";


    document.getElementById(
        "petal_length"
    ).value = "1.4";


    document.getElementById(
        "petal_width"
    ).value = "0.2";


    clearErrors();

}


/* =========================================================
   RESET
   ========================================================= */

function resetForm() {

    const form =
        document.getElementById(
            "predictionForm"
        );

    if (form) {
        form.reset();
    }

    clearErrors();


    const result =
        document.getElementById(
            "resultPanel"
        );

    if (result) {

        result.innerHTML = `

            <div class="result-placeholder">

                <div class="result-flower">
                    🌸
                </div>

                <h2>
                    Đang chờ phân tích
                </h2>

                <p>
                    Nhập thông số và nhấn
                    <strong>
                        PHÂN TÍCH BẰNG AI
                    </strong>
                    để bắt đầu.
                </p>

            </div>

        `;

    }

}


/* =========================================================
   PREDICT
   ========================================================= */

async function predictFlower(event) {

    event.preventDefault();


    if (!validateInputs()) {

        return;

    }


    const button =
        document.querySelector(
            ".predict-button"
        );


    const result =
        document.getElementById(
            "resultPanel"
        );


    const data = {

        sepal_length:
            getNumber("sepal_length"),

        sepal_width:
            getNumber("sepal_width"),

        petal_length:
            getNumber("petal_length"),

        petal_width:
            getNumber("petal_width")

    };


    button.disabled = true;

    button.textContent =
        "⏳ ĐANG PHÂN TÍCH...";


    result.innerHTML = `

        <div class="result-placeholder">

            <div class="result-flower">
                🧠
            </div>

            <h2>
                AI đang phân tích...
            </h2>

            <p>
                Mô hình SVM đang xử lý
                bốn đặc trưng của hoa.
            </p>

        </div>

    `;


    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(data)
                }
            );


        const resultData =
            await response.json();


        if (!response.ok) {

            throw new Error(
                resultData.message ||
                "Không thể dự đoán."
            );

        }


        showPrediction(
            resultData
        );

    }

    catch (error) {

        result.innerHTML = `

            <div class="result-placeholder">

                <div class="result-flower">
                    ⚠️
                </div>

                <h2>
                    Có lỗi xảy ra
                </h2>

                <p>
                    ${error.message}
                </p>

            </div>

        `;

    }

    finally {

        button.disabled = false;

        button.textContent =
            "✦ PHÂN TÍCH BẰNG AI";

    }

}


/* =========================================================
   SHOW RESULT
   ========================================================= */

function showPrediction(data) {

    const result =
        document.getElementById(
            "resultPanel"
        );


    const confidence =
        Number(data.confidence);


    result.innerHTML = `

        <div class="result-success">

            <img
                src="${data.image}"
                class="result-image"
                alt="${data.name_vi}"
                onerror="this.style.display='none'"
            >

            <span class="small-label">
                AI PREDICTION
            </span>

            <h2>
                ${data.name_vi}
            </h2>

            <p>
                ${data.description}
            </p>


            <div class="confidence">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:8px;
                ">

                    <span>
                        Độ tin cậy
                    </span>

                    <strong>
                        ${confidence}%
                    </strong>

                </div>


                <div class="confidence-bar">

                    <div
                        class="confidence-fill"
                        style="
                            width:${confidence}%
                        "
                    ></div>

                </div>

            </div>


            <div class="result-details">

                <div class="detail-box">

                    <small>
                        Tên khoa học
                    </small>

                    <strong>
                        ${data.name}
                    </strong>

                </div>


                <div class="detail-box">

                    <small>
                        Màu sắc
                    </small>

                    <strong>
                        ${data.color}
                    </strong>

                </div>


                <div class="detail-box">

                    <small>
                        Phân loại
                    </small>

                    <strong>
                        ${data.class}
                    </strong>

                </div>


                <div class="detail-box">

                    <small>
                        Nguồn gốc
                    </small>

                    <strong>
                        ${data.origin}
                    </strong>

                </div>

            </div>


            <div
                style="
                    margin-top:20px;
                    padding:18px;
                    border-radius:16px;
                    background:rgba(255,255,255,0.04);
                "
            >

                <strong>
                    Đặc điểm
                </strong>

                <p
                    style="
                        color:var(--muted);
                        margin-bottom:0;
                    "
                >
                    ${data.characteristic}
                </p>

            </div>

        </div>

    `;

}


/* =========================================================
   AUTO SAMPLE
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        const form =
            document.getElementById(
                "predictionForm"
            );

        if (form) {

            fillSample();

        }

    }
);