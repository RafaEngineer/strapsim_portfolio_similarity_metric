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
    require('../PDWEL-master/Database.php');
    $DataBase = new Database();
    $sql = "SELECT * FROM usuarios WHERE id > :id";
    $binds = ['id' =>2];
    $result = $DataBase->SELECT($sql, $binds);
    if($result ->rowCount() >0){
        $dados = $result ->fetchAll(PDO::FETCH_OBJ);
        foreach ($dados as $item){
            echo "<div class='result'>";
            echo "Nome : {$item->nome} <br>";
            echo "Email : {$item->email} <br>";
            echo "descrição : {$item->descricao} <br>";
            echo "</div>";
        }
    }
    ?>
</div>
</body>
</html>