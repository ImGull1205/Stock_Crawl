const companySelector = document.getElementById("company-selector");
companies.forEach((company) => {
    const option = document.createElement("option");
    option.value = company;
    option.textContent = company;
    companySelector.appendChild(option);
});

document.getElementById("company-selector").addEventListener("change", function () {
    const selectedCompany = this.value;
    if (selectedCompany) {
        fetchNewsForCompany(selectedCompany);
        fetchCompanyInfo(selectedCompany);
    }
});

function fetchNewsForCompany(company) {
    const newsContainer = document.getElementById("news-container");
    const newsApiUrl = `https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/${company}/activity-news`;

    newsContainer.innerHTML = "";

    function formatDate(dateString) {
        const options = { year: "numeric", month: "long", day: "numeric" };
        return new Date(dateString).toLocaleDateString(undefined, options);
    }

    fetch(newsApiUrl, { method: "GET" })
        .then((response) => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            return response.json();
        })
        .then((data) => {
            if (!data.listActivityNews || !Array.isArray(data.listActivityNews)) {
                throw new Error("Invalid response format");
            }

            data.listActivityNews.forEach((article) => {
                const newsCard = document.createElement("div");
                newsCard.className = "news-card";

                const newsTitle = document.createElement("div");
                newsTitle.className = "news-title";
                newsTitle.textContent = article.title;

                const newsSource = document.createElement("div");
                newsSource.className = "news-source";
                newsSource.textContent = article.source ? `Nguồn: ${article.source}` : "Nguồn: Không rõ";

                const newsDate = document.createElement("div");
                newsDate.className = "news-date";
                newsDate.textContent = `Ngày đăng: ${formatDate(article.publishDate)}`;

                const newsPrice = document.createElement("div");
                newsPrice.className = "news-price";
                newsPrice.textContent = article.price ? `Giá: ${article.price} VND` : "Giá: Không có thông tin";

                const newsChange = document.createElement("div");
                newsChange.className = `news-change${parseInt(article.priceChange) > 0 ? "" : "-minus"}`;
                newsChange.textContent =
                    article.priceChange !== null
                        ? `Thay đổi: ${article.priceChange} (${(article.priceChangeRatio * 100).toFixed(2)}%)`
                        : "Thay đổi: Không có thông tin";

                newsCard.appendChild(newsTitle);
                newsCard.appendChild(newsSource);
                newsCard.appendChild(newsDate);
                newsCard.appendChild(newsPrice);
                newsCard.appendChild(newsChange);

                newsContainer.appendChild(newsCard);
            });
        })
        .catch((error) => {
            console.error("Error fetching news:", error);
            newsContainer.textContent = `Không thể tải tin tức: ${error.message}`;
        });
}

function fetchCompanyInfo(company) {
    const companyInfoContainer = document.getElementById("company-info");
    const companyInfoApiUrl = `https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/${company}/overview`;

    // Xóa toàn bộ nội dung của companyInfoContainer trước khi thêm dữ liệu mới
    companyInfoContainer.innerHTML = "";

    fetch(companyInfoApiUrl, { method: "GET" })
        .then((response) => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            return response.json();
        })
        .then((data) => {
            // Kiểm tra dữ liệu có tồn tại không
            if (!data || typeof data !== "object") {
                throw new Error("Invalid response format");
            }

            const companyInfoCard = document.createElement("div");
            companyInfoCard.className = "company-info-card";

            const companyName = document.createElement("div");
            companyName.className = "company-name";
            companyName.textContent = `Tên công ty: ${data.shortName || "Không có thông tin"}`;

            const companyIndustry = document.createElement("div");
            companyIndustry.className = "company-industry";
            companyIndustry.textContent = `Ngành: ${data.industry || "Không có thông tin"}`;

            const companyEstablishedYear = document.createElement("div");
            companyEstablishedYear.className = "company-established-year";
            companyEstablishedYear.textContent = `Năm thành lập: ${data.establishedYear || "Không có thông tin"}`;

            const companyNoEmployees = document.createElement("div");
            companyNoEmployees.className = "company-no-employees";
            companyNoEmployees.textContent = `Số nhân viên: ${data.noEmployees || "Không có thông tin"}`;

            const companyWebsite = document.createElement("div");
            companyWebsite.className = "company-website";
            const websiteLink = document.createElement("a");
            websiteLink.href = data.website;
            websiteLink.target = "_blank";
            websiteLink.textContent = data.website || "Không có thông tin";
            companyWebsite.appendChild(websiteLink);

            companyInfoCard.appendChild(companyName);
            companyInfoCard.appendChild(companyIndustry);
            companyInfoCard.appendChild(companyEstablishedYear);
            companyInfoCard.appendChild(companyNoEmployees);
            companyInfoCard.appendChild(companyWebsite);

            companyInfoContainer.appendChild(companyInfoCard);
        })
        .catch((error) => {
            console.error("Error fetching company info:", error);
            companyInfoContainer.textContent = `Không thể tải thông tin công ty: ${error.message}`;
        });
}
