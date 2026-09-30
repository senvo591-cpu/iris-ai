async function predictFlower() {

    const sepalLength =
        document.getElementById("sepal_length").value;

    const sepalWidth =
        document.getElementById("sepal_width").value;

    const petalLength =
        document.getElementById("petal_length").value;

    const petalWidth =
        document.getElementById("petal_width").value;

    const error =
        document.getElementById("error");

    const result =
        document.getElementById("result");


    error.innerHTML = "";


    // ==============================
    // VALIDATION
    // ==============================

    const values = [
        sepalLength,
        sepalWidth,
        petalLength,
        petalWidth
    ];

    if (
        values.some(
            value =>
                value === "" ||
                Number(value) <= 0
        )
    ) {

        error.innerHTML =
            "⚠ Vui lòng nhập đầy đủ các giá trị lớn hơn 0.";

        return;
    }


    // ==============================
    // LOADING
    // ==============================

    result.innerHTML = `

        <div class="loading">

            <div class="loader"></div>

            <h2>
                AI đang phân tích...
            </h2>

            <p>
                SVM đang tìm kiếm mẫu phù hợp.
            </p>

        </div>

    `;


    // ==============================
    // SEND REQUEST
    // ==============================

    try {

        const response =
            await fetch("/api/predict", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    sepal_length:
                        Number(sepalLength),

                    sepal_width:
                        Number(sepalWidth),

                    petal_length:
                        Number(petalLength),

                    petal_width:
                        Number(petalWidth)

                })

            });


        if (!response.ok) {

            throw new Error(
                "Prediction failed"
            );

        }


        const data =
            await response.json();


        // ==============================
        // IMAGE
        // ==============================

        const image =
            `/static/images/${data.prediction}.jpg`;


        // ==============================
        // RESULT
        // ==============================

        result.innerHTML = `

            <div class="result-content">

                <div class="result-image-wrapper">

                    <img
                        src="${image}"
                        alt="${data.display_name}"
                        class="flower-result-image"
                    >

                </div>


                <div class="prediction-label">

                    ✦ PREDICTION

                </div>


                <h2>
                    ${data.display_name}
                </h2>


                <div class="confidence">

                    <div class="confidence-number">
                        ${data.confidence}%
                    </div>

                    <div>
                        SVM confidence
                    </div>

                </div>


                <div class="measurements">

                    <div>
                        <span>Sepal Length</span>
                        <strong>
                            ${sepalLength} cm
                        </strong>
                    </div>

                    <div>
                        <span>Sepal Width</span>
                        <strong>
                            ${sepalWidth} cm
                        </strong>
                    </div>

                    <div>
                        <span>Petal Length</span>
                        <strong>
                            ${petalLength} cm
                        </strong>
                    </div>

                    <div>
                        <span>Petal Width</span>
                        <strong>
                            ${petalWidth} cm
                        </strong>
                    </div>

                </div>

            </div>

        `;

    }


    catch (err) {

        result.innerHTML = `

            <div class="result-placeholder">

                <div class="big-flower">
                    ⚠️
                </div>

                <h2>
                    Không thể dự đoán
                </h2>

                <p>
                    Vui lòng kiểm tra server FastAPI.
                </p>

            </div>

        `;

    }

}


function fillSample() {

    document.getElementById(
        "sepal_length"
    ).value = 5.1;

    document.getElementById(
        "sepal_width"
    ).value = 3.5;

    document.getElementById(
        "petal_length"
    ).value = 1.4;

    document.getElementById(
        "petal_width"
    ).value = 0.2;

}