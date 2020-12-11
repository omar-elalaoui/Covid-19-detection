$("#predict").hide();
$("#image-selector").change(function () {
    $("#selected-image").show();
    let reader = new FileReader();
    reader.onload = function (e) {
        let dataURL = reader.result;
        $('#selected-image').attr("src", dataURL);
        base64Image = dataURL.replace("data:image/jpeg;base64,", "");
        base64Image = base64Image.replace("data:image/jpg;base64,", "");
        base64Image = base64Image.replace("data:image/png;base64,", "");
    };
    reader.readAsDataURL($("#image-selector")[0].files[0]);
    $("#predict").show();
});

$('.img_test_item').click(function () {
    $("#selected-image").show();
    $('#selected-image').attr("src", $(this).attr('src'));
    $("#predict").show();
});

$("#predict").click(function (event) {
    let message = {
        image: base64Image
    };
    $(".lds-dual-ring").show();
    console.log(message);
    $('.incV3_pos button').html('');
    $('.incV3_neg button').html('');
    $('.resNet50_pos button').html('');
    $('.resNet50_neg button').html('');
    $('.VGG16_pos button').html('');
    $('.VGG16_neg button').html('');
    $('.danseNet201_pos button').html('');
    $('.danseNet201_neg button').html('');

    $.post("http://34.70.202.170/predict", JSON.stringify(message), function (response) {
        let incV3 = response.inceptionV3;
        let resNet50 = response.resNet50;
        let VGG16 = response.VGG16;
        let danseNet201 = response.danseNet201;
        affiche(incV3, ".incV3_pos", ".incV3_neg");
        affiche(resNet50, ".resNet50_pos", ".resNet50_neg");
        affiche(VGG16, ".VGG16_pos", ".VGG16_neg");
        affiche(danseNet201, ".danseNet201_pos", ".danseNet201_neg");
        $(".lds-dual-ring").hide()
    });
});

function affiche(model, pos, neg) {
    if (model.class == "Covid") {
        $(pos + " button").html((model.value * 100).toFixed(2) + '%');
    } else {
        $(neg + " button").html((model.value * 100).toFixed(2) + '%');
    }
}
