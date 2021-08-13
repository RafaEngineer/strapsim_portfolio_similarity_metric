<!DOCTYPE html>
<html lang="pt-br">
<head>
    <title>Sistema Crud</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
<div class="crud">
   <h3>Lista de Usuário</h3>
    <?php
    require('../Udemy/Database.php');
    $DataBase = new Database();
    $sql = "SELECT * FROM usuarios WHERE id > :id";
    $binds = ['id' =>2];
    $dados = $DataBase->SELECT($sql, $binds);
    if($dados ->rowCount() >0){
        foreach ($dados as $item){
            echo "div class='result'>";
            echo "Nome : {$item['nome']} <br>";
            echo "Email : {$item['email']} <br>";
            echo "descrição : {$item['descricao']} <br>";
            echo "</div>";
        }
    }
    ?>
</div>
</body>
</html>